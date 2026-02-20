POLYCHROMATIC**

**LANGUAGE MODEL**

*State-Conditional Activation Functions Inspired by
Neurotransmitter-Receptor Diversity*

Project Technical Report

Version 1.1 \| February 2026

**Daniel Nobrega**

Independent Research

**1. Executive Summary**

This report presents **PolychromaticLM**, a novel neural network
architecture that replaces fixed activation functions in transformer
feed-forward blocks with **state-conditional, learnable activation
routing**. Inspired by the diversity of neurotransmitter-receptor
interactions in biological neural systems, the architecture introduces a
mechanism where each neuron in the network maintains a static preference
profile across multiple activation functions, modulated dynamically by
the global state of the layer. This mirrors how biological neurons
express specific receptor subtypes (static) while being modulated by
fluctuating neurotransmitter concentrations in the synaptic cleft
(dynamic).

The core hypothesis is that the computational capacity of a neural
network scales not merely with parameter count, but with the
**informational richness per connection**. In current architectures,
each weight is a scalar (float16/32). In the biological brain, each
synapse is a multidimensional object whose behavior depends on the
chemical context. By approximating this property, we expect to achieve
greater expressive power per parameter, enabling a 600M-parameter model
to exhibit capabilities typically associated with larger networks.

The project proceeds in two phases: **Phase 1** validates the mechanism
on mathematical reasoning tasks, comparing against established
baselines. **Phase 2** (future work) applies the architecture to a novel
synthetic consciousness dataset to test emergent properties.

**2. Motivation and Biological Basis**

**2.1 The Combinatorial Argument**

The human brain contains approximately 86 billion neurons with an
estimated 10¹⁵ synapses. In a naive binary model (synapse on/off), the
state space is 2\^(10¹⁵). However, biological synapses are not binary
switches. Each synaptic connection involves a specific combination of
neurotransmitter type, receptor subtype, phosphorylation state, ionic
context, and plasticity strength. Conservative estimates suggest \~100
functionally distinct states per synapse, yielding a state space of:

> *Ω = 100\^(10¹⁵) = 10\^(2 × 10¹⁵)*

This represents a difference in the **exponent of the exponent**
compared to the binary model. This suggests that the brain's
extraordinary computational capacity may arise not from neuron count
alone, but from the qualitative diversity of each connection.

**2.2 The Analogy to Artificial Neural Networks**

In standard transformers, every weight is a scalar and every activation
function is fixed (typically SiLU, GELU, or ReLU). This is analogous to
a brain where every synapse uses only one neurotransmitter and one
receptor type. The PolychromaticLM introduces a mechanism that allows
each neuron to select from multiple qualitatively different activation
functions, conditioned on the network state, thereby enriching the
informational content of each connection.

The key insight: if the state space of a network scales as Cˢ (where C =
complexity per connection, S = number of connections), then increasing C
even modestly yields exponential gains, potentially more efficiently
than increasing S (adding parameters).

**2.3 Neurotransmitter-Activation Mapping**

We map four principal neurotransmitter systems to four qualitatively
distinct activation functions, chosen to cover a broad range of
mathematical behaviors:

  ---------------------- -------------- ---------------- ------------------------------------
  **Neurotransmitter**   **Biological   **Activation**   **Mathematical Behavior**
                         Role**                          

  Glutamate              Fast           ReLU             Hard threshold, passes positive
                         excitation                      signals, blocks negative. Sparse,
                                                         binary gating.

  GABA                   Inhibition     Tanh             Symmetric compression to \[-1,1\].
                                                         Suppresses extreme values. Bounded
                                                         output.

  Dopamine               Reward /       SiLU (Swish)     Self-gated: x·σ(x). Smooth, allows
                         modulation                      mild negatives. Rich gradient
                                                         landscape.

  Acetylcholine          Attention /    GELU             Probabilistic gating: x·Φ(x). Soft,
                         learning                        attention-like activation.
  ---------------------- -------------- ---------------- ------------------------------------

*These four functions are qualitatively distinct (not parametric
variations of each other), maximizing the space of transformations
available to the routing mechanism.*

**3. Related Work**

**3.1 Kolmogorov-Arnold Networks (KAN)**

KANs (Liu et al., 2024) are the most directly related work. They replace
fixed activation functions on nodes with learnable activation functions
(B-splines) on edges. Every weight parameter becomes a univariate
function. KANs have shown that smaller networks with richer connections
can match or outperform larger MLPs. Our work shares the core insight
(enrich the connection, not just scale the network) but differs in
mechanism: KANs learn a single function per edge, while PolychromaticLM
routes between qualitatively different functions conditioned on network
state.

**3.2 Learnable Activation Selection via Gumbel-Softmax**

Recent work has used the Gumbel-Softmax trick to enable differentiable
selection among predefined activation functions during training.
However, existing approaches treat this selection as input-independent
(the same activation is selected regardless of context). PolychromaticLM
extends this by making the selection state-conditional, where the choice
depends on the hidden representation flowing through the network.

**3.3 Dynamic Neural Networks**

The field of dynamic neural networks encompasses architectures that
adapt structure or parameters to different inputs, including Mixture of
Experts (MoE), early exiting, and dynamic routing. PolychromaticLM can
be seen as a fine-grained MoE at the activation function level, where
each neuron routes to different "expert" transformations.

**3.4 Biologically-Inspired Synaptic Diversity**

A 2025 paper in Nature Communications implemented "weight splitting"
(multiple connections between neuron pairs with varying weights),
inspired by biological multi-synaptic connectivity. This represents the
simplest version of the idea. PolychromaticLM goes further by varying
the qualitative nature of each connection, not just its strength.

**3.5 Trainable Activation Functions (DiTAC, PReLU, ACON)**

Various approaches introduce trainable parameters into activation
functions (e.g., PReLU's learnable slope, DiTAC's diffeomorphic
transformations). These enrich individual activations but maintain a
single function type. PolychromaticLM maintains multiple distinct
function types and routes between them.

**4. Architecture**

**4.1 Base Model: Qwen3-0.6B Skeleton**

We reimplement the Qwen3-0.6B architecture from scratch in PyTorch as
our base transformer, using the same tokenizer (byte-level BPE,
vocabulary size 151,669). The base specifications are:

  ----------------------------------- -----------------------------------
  **Parameter**                       **Value**

  Total Parameters                    \~600M (+ \~1.4M for routing,
                                      \~0.23%)

  Non-Embedding Parameters            \~440M

  Hidden Dimension (d_model)          1,024

  FFN Intermediate Dimension (d_ff)   4,096

  Number of Layers                    28

  Attention Heads (Query)             16

  Attention Heads (Key/Value)         8 (Grouped Query Attention)

  Head Dimension                      64

  Position Encoding                   RoPE (Rotary Position Embeddings)

  Normalization                       RMSNorm (pre-norm) with QK-Norm

  FFN Activation                      PolychromaticGLU (see Section 4.2)

  Embedding Tying                     Yes (input/output share weights)

  Context Length                      4,096 tokens (Phase 1)

  Precision                           BFloat16 mixed precision
  ----------------------------------- -----------------------------------

**4.2 PolychromaticGLU: The Core Innovation**

The standard SwiGLU block in transformer FFNs computes:

> *SwiGLU(x) = SiLU(x · W_gate) ⊙ (x · W_up)*

Where SiLU is a fixed activation function applied uniformly. We replace
this with PolychromaticGLU:

> *PolyGLU(x) = \[∑\_{k=1}\^{K} g_k\^(i)(h) · σ_k(x · W_gate)\] ⊙ (x ·
> W_up)*

Where K=4 activation candidates (σ_k), and the gating weights g_k are
produced by a hybrid static-dynamic routing mechanism described below.

**4.3 Hybrid Routing Mechanism**

The routing mechanism combines per-neuron static preferences with
per-layer dynamic modulation, directly analogous to biological
neurotransmission:

> *g_k\^(i) = GumbelSoftmax(α_k\^(i) + β_k · f(h), τ)*

**4.3.1 Static Component: Receptor Profile (α)**

Each of the 4,096 neurons in the FFN intermediate layer maintains a
learnable preference vector α ∈ R⁴ over the K=4 activation functions.
These are analogous to the receptor distribution on a neuronal membrane,
which is relatively stable over time. During training, these preferences
are learned via backpropagation. Shape per layer: \[4096, 4\] = 16,384
parameters.

**4.3.2 Dynamic Component: Neurotransmitter Concentration (β · f(h))**

A small gating network receives the layer's hidden state (mean-pooled
across the sequence dimension) and produces K=4 modulation signals.
These are analogous to the concentration of neurotransmitters in the
synaptic cleft, which varies with context. Architecture of the gating
network:

> *f(h) = Linear_2(ReLU(Linear_1(mean_pool(h))))*

Where Linear_1: 1024 → 32 and Linear_2: 32 → 4. Parameters per layer:
\~33K. The output β_k · f(h) is broadcast to all 4,096 neurons, shifting
their static preferences toward whichever activation is contextually
appropriate.

**4.3.3 Temperature Annealing**

During training, the Gumbel-Softmax temperature τ is annealed from 1.0
to 0.1 over the course of training. High temperature allows exploration
(soft mixture of all activations); low temperature forces commitment
(near-hard selection). During inference, we use argmax on the logits for
zero overhead beyond the gating network computation.

**4.3.4 Parameter and Compute Overhead**

  --------------------- ------------------ ----------------- ----------------
  **Component**         **Params/Layer**   **Total (28       **% of Model**
                                           layers)**         

  Static preferences    16,384             458,752           0.076%
  (α)                                                        

  Gating network (f)    \~33,000           \~924,000         0.154%

  **Total routing       **\~49,384**       **\~1,382,752**   **\~0.23%**
  overhead**                                                 
  --------------------- ------------------ ----------------- ----------------

Compute overhead during training: \~15-25% (computing 4 activations in
parallel). During inference with hard selection: negligible beyond the
gating network forward pass.

**5. Experimental Plan**

**5.1 Phase 1: Mathematical Reasoning (Current)**

**5.1.1 Objective**

Validate that PolychromaticGLU improves parameter efficiency and
mathematical reasoning capability compared to standard SwiGLU.
Demonstrate that enriching connection quality (C) yields gains analogous
to increasing connection quantity (S).

**5.1.2 Training Data**

***Updated in v1.1.*** We adopt NVIDIA's Nemotron pre-training dataset
family as our primary data source, replacing the previously planned mix
of OpenWebMath and proof-pile-2. The Nemotron datasets offer superior
quality due to their Lynx + LLM cleaning pipeline, which preserves
mathematical structure (equations, code blocks, LaTeX notation) during
HTML-to-text conversion, followed by quality filtering and global
deduplication.

The pre-training data mix targets \~10B tokens with the following
composition:

  --------------------- ----------- ------------ ----------------------------------
  **Dataset**           **Share**   **Tokens**   **Role**

  Nemotron-CC-Math-v1   70%         \~7B         Core mathematics: high-quality
  (4+ subset)                                    math-rich web content with
                                                 preserved LaTeX notation,
                                                 equations, and proofs.
                                                 Quality-filtered (scores 4-5) and
                                                 deduplicated.

  Nemotron-CC-v2 / v2.1 25%         \~2.5B       General STEM: broad English web
                                                 content including science,
                                                 reasoning, and natural language
                                                 prose. Provides robustness and
                                                 linguistic diversity to prevent
                                                 overfitting on pure math notation.

  Nemotron-CC-Code-v1   5%          \~0.5B       Code: Python, mathematical
                                                 libraries (NumPy, SageMath). Acts
                                                 as regularizer for structured
                                                 sequential reasoning throughout
                                                 training.
  --------------------- ----------- ------------ ----------------------------------

**Data mix annealing:** Following recent findings on the importance of
data quality scheduling (cf. Llama 3 annealing strategy), we implement a
data mix annealing schedule over the final 20% of training. During the
first 80% of training, proportions remain at the baseline mix above.
During the annealing phase (80--100% of training), the high-quality math
proportion increases from 70% to \~85%, while STEM vanilla decreases
from 25% to \~10%. Code proportion remains constant at 5% throughout,
serving as a distributed regularizer for structured reasoning.

**Rationale for dataset selection:** Nemotron-CC-Math-v1 was chosen over
alternatives (OpenWebMath, FineMath, proof-pile-2) based on three
factors: (1) pipeline quality --- the Lynx rendering + Phi-4 LLM
cleaning pipeline recovers math across diverse HTML formats (MathJax,
KaTeX, MathML) that brittle heuristic parsers miss; (2) scale --- the 4+
subset alone provides 52B tokens, ensuring we sample without repetition
for our 7B token allocation; (3) benchmark validation --- NVIDIA
reported +4.8 to +12.6 points on MATH benchmarks over previous best
datasets when using Nemotron-CC-Math for pre-training.

**Tokenization:** All data is tokenized using the Qwen3 tokenizer
(byte-level BPE, vocabulary size 151,669), reused as-is from
HuggingFace. Pre-training data is stored as concatenated uint32 arrays
in binary chunks (\~100M tokens each), with \<EOS\> tokens separating
documents. The dataloader implements intra-document causal attention
masking (document masking) via Flash Attention 2 to prevent
cross-document attention leakage within training windows.

**Post-training (SFT):** Supervised fine-tuning uses Nemotron-Math-v2, a
dataset of \~347K high-quality mathematical problems with 7M
model-generated reasoning trajectories produced by gpt-oss-120b under
multiple reasoning modes. Answers are verified via LLM-as-a-judge and
filtered by pass-rate. SFT is conducted for 1--2 epochs (\~4--6 hours on
A100), with loss computed only on assistant response tokens (user prompt
tokens are masked). Additional SFT runs with complementary datasets
(MetaMathQA, NuminaMath-CoT) may be conducted if initial results warrant
exploration of alternative solution styles.

**5.1.3 Training Configuration**

  ----------------------------------- -----------------------------------
  **Hyperparameter**                  **Value**

  Optimizer                           AdamW (β1=0.9, β2=0.95, ε=1e-8)

  Peak Learning Rate                  1e-4 (cosine decay with warmup)

  Warmup Steps                        2,000

  Weight Decay                        0.1

  Batch Size (effective)              \~512K tokens (micro-batch × grad
                                      accumulation)

  Precision                           BFloat16 mixed precision

  Gradient Clipping                   1.0

  Gumbel-Softmax τ schedule           1.0 → 0.1 (linear annealing over
                                      full training)

  Training Tokens                     \~10B

  Estimated Training Time             \~85--90 hours on A100 80GB
  ----------------------------------- -----------------------------------

**5.1.4 Evaluation Benchmarks**

  --------------- --------------------------- ---------------------------
  **Benchmark**   **What It Measures**        **Why It Matters**

  GSM8K           Grade school math reasoning Multi-step reasoning with
                                              arithmetic

  MATH-500        Competition math problems   Higher-order symbolic
                                              reasoning

  MMLU (STEM)     Broad STEM knowledge        Generalization beyond pure
                                              math

  Perplexity      Language modeling quality   Baseline metric for model
                                              quality
  --------------- --------------------------- ---------------------------

**5.1.5 Baselines**

Priority-ordered comparison strategy:

**Primary (controlled):** If budget permits, train VanillaLM-0.6B
(identical architecture with standard SwiGLU) on the same data for the
same number of tokens. This is the most scientifically rigorous
comparison.

**External references:** Compare against Qwen3-0.6B-Base (ceiling,
trained on 36T tokens), Pythia-410M, SmolLM-360M/SmolLM2-360M, and
Gemma-2-2B (larger but contextualizes where our model stands).

**5.1.6 Ablation Studies**

If compute permits, the following ablations provide insight into
architectural choices:

**K={2, 4, 6}:** Test with 2 activations (ReLU + Tanh), 4 (full set),
and 6 (adding Sine + Gaussian) to characterize the marginal value of
increasing activation diversity.

**Routing granularity:** Compare per-layer-only routing (no static α
preferences) versus the full hybrid mechanism to quantify the value of
per-neuron specialization.

**Temperature schedule:** Compare different τ annealing schedules and
their effect on final performance and routing diversity.

**5.2 Phase 2: Synthetic Consciousness (Future Work)**

**5.2.1 Concept**

Generate a large-scale synthetic dataset simulating the internal mental
life of a fictional person: episodic memories, semantic knowledge,
internal monologue, emotional associations, reconstructed dialogues,
dreams, and deliberate confabulations. The dataset would model memory as
the brain actually stores it: fragmentary, overlapping, temporally
distorted, and emotionally colored. Train PolychromaticLM on this data
and evaluate whether the enriched activation routing produces more
coherent, human-like associative behavior compared to standard
architectures.

**5.2.2 Evaluation Criteria**

**Associative coherence:** Given a memory fragment, does the model
generate associations that are contextually consistent with the
simulated person's life history?

**Identity continuity:** Does the model maintain a consistent
"personality" across diverse prompts?

**Reduced Turing tests:** Can human evaluators distinguish
PolychromaticLM outputs from vanilla baseline outputs on tasks requiring
"human-like" responses?

**Overlapping context perplexity:** Performance on sequences that
require simultaneous maintenance of multiple temporal/emotional
contexts.

**6. Infrastructure and Budget**

**6.1 Compute Resources**

***Updated in v1.1.*** Budget revised to reflect SFT time allocation
based on Nemotron-Math-v2 dataset.

  ------------------ -------------- ---------- ----------------- -------------------
  **Item**           **Platform**   **GPU**    **Est. Hours**    **Est. Cost**

  PolychromaticLM    RunPod         A100 80GB  85--90h           \~\$148
  pre-training                                                   

  PolychromaticLM    RunPod         A100 80GB  4--6h             \~\$10
  SFT                                                            

  VanillaLM training RunPod         A100 80GB  65--70h           \~\$115
  (if budget)                                                    

  Debugging /        RunPod         A100 80GB  \~35h             \~\$57
  ablations                                                      

  Evaluation runs    RunPod         A100 80GB  \~10h             \~\$16

  **TOTAL**                                    **\~200--210h**   **\~\$346**
  ------------------ -------------- ---------- ----------------- -------------------

*Budget: \~R\$2,000 (\~US\$364). Estimated utilization: \~\$346, leaving
\~\$18 buffer. A100 80GB at \~\$1.64/hr (RunPod community cloud).
Priority: PolychromaticLM first; VanillaLM baseline only if budget
permits. Data tokenization runs on Google Colab (free tier) and does not
require GPU.*

**6.2 Software Stack**

PyTorch 2.x (model implementation from scratch), Flash Attention 2
(efficient attention computation with document masking support),
DeepSpeed ZeRO Stage 2 or FSDP (memory optimization), Weights & Biases
(experiment tracking), lm-evaluation-harness (EleutherAI benchmark
suite), Qwen3 tokenizer (HuggingFace, reused as-is), HuggingFace
datasets library (data download and streaming).

**7. Interpretability and Analysis**

One of the most compelling aspects of PolychromaticLM is its inherent
interpretability. Post-training, we can extract and visualize:

**7.1 Neurotransmitter Maps**

For each layer, visualize which activation function each neuron
"prefers" (argmax of learned α vectors). This produces a
neurotransmitter map of the network, showing the distribution of
glutamate-like, GABA-like, dopamine-like, and acetylcholine-like neurons
across layers. Hypothesis: early layers will skew toward ReLU (fast
feature extraction), middle layers toward GELU/SiLU (modulation and
attention), and final layers toward Tanh (output compression).

**7.2 Dynamic Routing Analysis**

For different types of inputs (e.g., arithmetic vs. algebra vs.
geometry), track how the dynamic modulation shifts neurotransmitter
balance. This shows whether the network learns to engage different
"cognitive modes" for different problem types, analogous to how
biological neurotransmitter levels fluctuate with cognitive demands.

**7.3 Activation Diversity Metrics**

Quantify the entropy of activation selections across neurons and layers.
High entropy = diverse, overlapping representations (the biological
ideal). Low entropy = convergence to a single activation (equivalent to
standard SwiGLU, indicating the mechanism is not being used). This
metric directly tests whether the model actually leverages the
polychromatic capacity.

**8. Risks and Mitigations**

  ----------------------- ----------------------- -----------------------
  **Risk**                **Impact**              **Mitigation**

  Model collapses to      PolychromaticGLU        Entropy regularization
  single activation (low  reduces to standard     loss; diversity reward
  diversity)              SwiGLU, nullifying the  in routing
                          contribution            

  Compute overhead        Insufficient training   Hard activation
  exceeds budget          tokens                  selection (no parallel
                                                  compute of all K);
                                                  reduce K to 2

  Training instability    Loss spikes, NaN        Careful temperature
  from Gumbel-Softmax     gradients               schedule; gradient
                                                  clipping; separate LR
                                                  for routing params

  Unfair comparison       Reviewers question      Always report tokens
  (different data/tokens  validity                trained; prioritize
  vs baselines)                                   controlled VanillaLM
                                                  comparison
  ----------------------- ----------------------- -----------------------

**9. Project Timeline**

  ---------- --------------------- ----------------------------------------
  **Week**   **Milestone**         **Deliverables**

  1--2       Implementation        Transformer skeleton + PolychromaticGLU
                                   in PyTorch; unit tests; small-scale
                                   validation

  3          Data pipeline         Nemotron dataset download, tokenization
                                   with Qwen3 tokenizer, binary chunk
                                   generation, streaming dataloader with
                                   document masking; debug training on
                                   small subset

  4--5       PolychromaticLM       Full pre-training run (\~85--90h A100)
             training              with data mix annealing; checkpointing;
                                   WandB monitoring. SFT on
                                   Nemotron-Math-v2 (\~4--6h)

  6          Evaluation +          Benchmark evaluation; VanillaLM training
             VanillaLM             if budget allows; ablations

  7--8       Analysis + writeup    Interpretability analysis;
                                   neurotransmitter maps; paper draft
  ---------- --------------------- ----------------------------------------

**10. Publication Strategy**

**10.1 Target Venues**

**Primary:** NeurIPS or ICML workshop on biologically-inspired AI,
neuro-AI, or efficient architectures.

**Secondary:** ICLR main conference (if results are strong enough for a
full paper).

**Preprint:** arXiv (cs.LG + cs.NE + q-bio.NC) for immediate visibility.

**10.2 Narrative Arc**

The paper tells a story from neuroscience to engineering: biological
observation (synaptic diversity creates computational richness) leads to
architectural insight (enrich connections, not just scale parameters),
which produces a concrete mechanism (PolychromaticGLU), validated on
established benchmarks. The interpretability analysis (neurotransmitter
maps) provides a visually compelling and scientifically meaningful
bridge back to the biological motivation.

**11. Conclusion**

PolychromaticLM proposes a simple but principled modification to
transformer architectures: replace fixed activation functions with
state-conditional, learnable activation routing, inspired by the
diversity of neurotransmitter-receptor interactions in the brain. The
mechanism adds only \~0.23% parameters and is designed to be a drop-in
replacement for standard SwiGLU blocks. By enriching the informational
content of each connection rather than scaling parameter count, we
hypothesize that PolychromaticLM can achieve greater expressive power
per parameter, offering a new axis for neural network scaling that is
orthogonal to the current paradigm of "more parameters, more data."

If the Phase 1 mathematical validation succeeds, Phase 2 will explore
whether this biologically-inspired enrichment enables qualitatively new
emergent behaviors when trained on data that mirrors the structure of
human consciousness: fragmented, overlapping, and emotionally textured.
The name "Polychromatic" reflects this vision: a network that sees the
world not in black and white scalars, but in the full spectrum of
synaptic color.