# Research Log: Routing Entropy Diagnostic Bug, Mid-Training Hotfix, and Emergent Specialization Discovery

**Date:** 2026-02-27
**Author:** Daniel Nobrega
**Project:** PolychromaticLM (600M parameter transformer with PolyGLU)
**Training stage:** Pre-training, step ~10,000 of ~19,531 (~51% complete)

---

## 1. Background

PolychromaticLM replaces the standard SwiGLU feed-forward block with PolyGLU (Polychromatic Gated Linear Unit), a novel mechanism that routes each neuron's activation through one of four candidate functions — ReLU, Tanh, SiLU, and GELU — inspired by neurotransmitter-receptor diversity in biological neural systems.

The routing decision is made via a two-component system:

```
logits = α + β · gate_net(mean_pool(h))
g_k = GumbelSoftmax(logits, τ)
```

Where:
- **α** ∈ ℝ^{d_ff × K} is a learned static per-neuron preference over the K=4 activations
- **gate_net** is a small MLP (d_model → 32 → K) that produces a dynamic, input-conditioned routing signal
- **β** ∈ ℝ^K scales the dynamic component
- **τ** is a temperature parameter annealed from 1.0 to 0.1 over training

The design intention was for α to capture stable per-neuron specialization (analogous to a neuron's baseline neurotransmitter identity) while gate_net provides context-dependent modulation (analogous to neuromodulatory state changes).

## 2. The Diagnostic Bug

### 2.1 What we were measuring

Since the start of training, routing entropy was logged to Weights & Biases every 100 steps using the following function:

```python
def get_routing_entropy(model):
    for i, block in enumerate(model.model_core):
        alpha = block.polyglu.alpha          # [d_ff, K]
        probs = softmax(alpha, dim=-1)       # [d_ff, K]
        entropy = -(probs * log(probs)).sum(dim=-1)  # [d_ff]
        # log entropy.mean() per layer
```

This computes entropy from `softmax(α)` alone — the **static component only**. It entirely ignores the dynamic routing signal `β · gate_net(h)`, which requires actual input data to compute.

### 2.2 What the metric showed

Over ~12,000 steps of training, routing entropy per layer remained essentially flat at ln(4) ≈ 1.38629, with deviations on the order of 10⁻⁵. This suggested the routing mechanism was not specializing — that all four activations were being used with near-uniform probability, effectively reducing PolyGLU to an expensive average of four activation functions.

### 2.3 Why it was misleading

The static-only metric was blind to the possibility that the dynamic component was doing all the routing work. Since α was initialized at zero and (as we discovered) was subject to weight decay pulling it back toward zero, the static softmax(α) would always be near-uniform regardless of what gate_net learned.

## 3. The Weight Decay Bug

### 3.1 Discovery

Upon investigating why α remained near zero, we identified that the optimizer parameter grouping used dimensionality as the sole criterion:

```python
for name, param in model.named_parameters():
    if param.ndim == 1:
        no_decay_params.append(param)    # weight_decay = 0.0
    else:
        decay_params.append(param)       # weight_decay = 0.1
```

The α parameter has shape [d_ff, K] = [4096, 4], which is 2-dimensional. It was therefore placed in the decay group and subjected to L2 regularization with λ=0.1 at every optimizer step. This continuously penalized any deviation of α from zero, actively suppressing static specialization.

### 3.2 Why this matters

In standard transformers, 2D parameters are weight matrices (e.g., W_q, W_k, W_v, W_up, W_down), and weight decay on these is standard practice for regularization. However, α is not a weight matrix — it is a routing preference parameter. Applying weight decay to α is analogous to applying L2 regularization to a temperature or bias term: it systematically prevents the parameter from learning its intended role.

### 3.3 Fix applied

The parameter grouping was corrected to explicitly route α to the no-decay group:

```python
for name, param in model.named_parameters():
    if param.ndim == 1:
        no_decay_params.append(param)
    elif 'alpha' in name:
        no_decay_params.append(param)    # routing param, not a weight matrix
    else:
        decay_params.append(param)
```

## 4. Mid-Training Intervention

### 4.1 Constraints

Training was ~51% complete at step ~10,000. Restarting from scratch was not feasible due to budget constraints (~$256 already spent on A100 compute). The fix had to be applied mid-training without destabilizing the model.

### 4.2 Procedure

Training was resumed from a portable checkpoint (`portable_step10000.pt`) rather than the DeepSpeed checkpoint. This was necessary because changing the parameter group membership (moving α from group 0 to group 1) would cause a shape mismatch in DeepSpeed's saved optimizer state.

**First attempt (failed):** We initially skipped loading the optimizer state entirely, creating a fresh AdamW. This caused the training loss to spike from ~2.0 to ~9.8 within the first gradient accumulation cycle. The reason: Adam's running variance estimate (`exp_avg_sq`) starts at zero with a fresh optimizer, causing the effective step size to be extremely large before the variance stabilizes — regardless of the nominal learning rate.

**Second attempt (successful):** We implemented an optimizer state transplant function that reconstructs the old parameter ordering, maps checkpoint state indices to parameter tensor identities, and copies per-parameter Adam states (exp_avg, exp_avg_sq, step) into the new optimizer. This preserves all accumulated optimizer statistics while only changing the weight decay group assignment for α. Training resumed cleanly with loss at ~2.05, consistent with the pre-intervention trajectory.

### 4.3 Verification

After the fix, the first three logged steps showed:

| Step  | Loss  | LR       | τ     | Static Entropy | Dynamic Entropy |
|-------|-------|----------|-------|----------------|-----------------|
| 10010 | 2.051 | 0.000057 | 0.539 | 1.386288       | 0.011096        |
| 10020 | 2.099 | 0.000057 | 0.538 | 1.386288       | 0.009786        |
| 10030 | 2.187 | 0.000057 | 0.538 | 1.386288       | 0.008016        |

Loss remained in the expected range with no spike, confirming the optimizer state transplant worked correctly.

## 5. Discovery: Emergent Deterministic Routing

### 5.1 The new metric

A complementary diagnostic function was added that measures routing entropy from the **full** routing logits (`α + β · gate_net(h)`) on real data, using forward hooks during an evaluation forward pass:

```python
def get_dynamic_routing_entropy(model, sample_input):
    # Registers hooks on each PolyGLU that capture:
    #   logits = alpha + beta * gate_net(mean_pool(x))
    #   probs = softmax(logits)
    #   entropy = -(probs * log(probs)).sum(dim=-1)
    # Returns per-layer and mean entropy
```

### 5.2 Results

At step 10,030, the per-layer dynamic routing entropy revealed:

| Layer | Dynamic Entropy | Interpretation         |
|-------|-----------------|------------------------|
| 0     | 0.000003        | Near-deterministic     |
| 1     | 0.000002        | Near-deterministic     |
| 2     | 0.000002        | Near-deterministic     |
| 3     | 0.000055        | Near-deterministic     |
| 4     | 0.000139        | Near-deterministic     |
| 5     | 0.000087        | Near-deterministic     |
| 6     | 0.000029        | Near-deterministic     |
| 7     | 0.000004        | Near-deterministic     |
| 8     | 0.000014        | Near-deterministic     |
| 9     | 0.183973        | Partially specialized  |
| 10    | 0.000121        | Near-deterministic     |
| 11    | 0.000727        | Near-deterministic     |
| 12    | 0.000566        | Near-deterministic     |
| 13    | 0.002424        | Mostly specialized     |
| 14    | 0.000502        | Near-deterministic     |
| 15    | 0.001079        | Near-deterministic     |
| 16    | 0.032677        | Mostly specialized     |
| 17    | 0.000691        | Near-deterministic     |
| 18    | 0.000145        | Near-deterministic     |
| 19    | 0.000071        | Near-deterministic     |
| 20    | 0.000125        | Near-deterministic     |
| 21    | 0.000208        | Near-deterministic     |
| 22    | 0.000064        | Near-deterministic     |
| 23    | 0.000030        | Near-deterministic     |
| 24    | 0.000095        | Near-deterministic     |
| 25    | 0.000124        | Near-deterministic     |
| 26    | 0.000077        | Near-deterministic     |
| 27    | 0.000401        | Near-deterministic     |
| **Mean** | **0.008016** | **Near-deterministic** |

For reference, ln(4) ≈ 1.3863 corresponds to a uniform distribution over 4 activations. The observed mean dynamic entropy of ~0.008 is approximately **0.6% of maximum**, indicating that the routing mechanism is making near-one-hot selections for the vast majority of neurons across all layers.

### 5.3 Interpretation

This is **emergent behavior** — no explicit loss term or regularizer was applied to encourage deterministic routing. The Gumbel-Softmax temperature at this training stage is τ ≈ 0.54, which still introduces substantial stochasticity. Despite this, the model learned routing logits with such high confidence that the softmax output is near-deterministic even before τ is fully annealed.

The key observations are:

1. **Dynamic routing dominates.** The gate_net (a 2-layer MLP processing mean-pooled hidden states) learned to produce strong, context-dependent routing signals. The static α component remained near zero (due to the weight decay bug), yet the routing mechanism still achieved extreme specialization through the dynamic pathway alone.

2. **Layer-dependent specialization depth.** Most layers (0–8, 10–12, 17–27) show entropy < 0.001, indicating near-perfect one-hot routing. Layers 9 and 16 show higher entropy (0.18 and 0.03 respectively), suggesting these layers benefit from mixing multiple activations for certain inputs. This heterogeneity across layers is itself an emergent property.

3. **Biological analogy holds.** In biological neural systems, neurotransmitter selection is not uniform or random — it is highly deterministic for a given neuronal context, with the specific neurotransmitter-receptor pairing depending on the circuit state. The gate_net has learned an analogous behavior: given the context (mean-pooled hidden state), it makes a confident, near-deterministic selection of which "neurotransmitter" (activation function) to apply per neuron.

### 5.4 Caveats and open questions

- **Performance validation pending.** Emergent specialization does not guarantee improved task performance. Ablation against a vanilla SwiGLU baseline is required to establish whether this routing behavior translates to measurable gains on downstream benchmarks (GSM8K, MATH-500).

- **Activation distribution unknown.** We have not yet analyzed *which* activations are being selected for *which* types of input. A qualitative analysis mapping activation preferences to token/context types would strengthen the biological analogy and provide interpretability insights.

- **α dynamics post-fix.** With weight decay removed from α, it remains to be seen whether α will begin to develop static preferences that complement the dynamic signal, or whether gate_net has already captured all useful routing information.

- **Single sample measurement.** The dynamic entropy was measured on whatever batch happened to be current at logging time. A more robust measurement would average over multiple diverse batches. However, the near-zero values across all layers suggest this is not a sampling artifact.

## 6. Files Modified

| File | Change | Purpose |
|------|--------|---------|
| `src/training/train.py` | `create_optimizer`: added `'alpha' in name` check | Remove weight decay from routing params |
| `src/training/train.py` | Added `_transplant_optimizer_state` function | Safe optimizer state migration across param groups |
| `src/training/train.py` | Portable checkpoint resume path updated | Use transplant instead of fresh optimizer |
| `src/training/train.py` | Logging block: added dynamic entropy call | Log full routing entropy alongside static |
| `src/model/model.py` | Added `get_dynamic_routing_entropy` function | Measure entropy from complete routing logits |
| `configs/train_config.yaml` | Added `resume_from` field | Point to portable checkpoint for resume |

No changes were made to `src/model/architecture.py` (frozen scientific contribution).

## 7. Timeline

| Time (UTC) | Event |
|------------|-------|
| ~Feb 21 02:25 | Training started (run dulcet-cosmos-1, multiple early crashes) |
| ~Feb 21 | Stable training begins, reaches step ~12,500 over 6.5 days |
| Feb 27 ~09:35 | Routing entropy graphs examined; plateau at ln(4) observed |
| Feb 27 ~10:00 | Diagnosis: static-only metric + weight decay bug identified |
| Feb 27 ~10:11 | First resume attempt (fresh optimizer) — loss spikes to ~9.8 |
| Feb 27 ~13:09 | Process killed; optimizer transplant approach implemented |
| Feb 27 ~13:55 | Second resume (toasty-durian-22) — loss stable at ~2.05 |
| Feb 27 ~14:00 | Dynamic entropy metric reveals near-deterministic routing |