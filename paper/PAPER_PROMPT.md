# Prompt: Write the PolychromaticLM arXiv Paper

You are tasked with writing a complete academic paper for arXiv submission about PolychromaticLM. The paper directory is `/workspace/PolyGLU/paper/`. All source materials are in `/workspace/PolyGLU/paper_reporting/`.

---

## Step 0: Setup

Create the directory structure for arXiv submission:

```
paper/
  main.tex          # The paper
  references.bib    # BibTeX references
  figures/          # Copy of all figures from paper_reporting/figures/
```

Copy ALL figures from `paper_reporting/figures/` into `paper/figures/`. These are the ONLY figures available — do not reference figures that don't exist.

---

## Step 1: Read ALL Source Materials

Before writing anything, read these files completely — they contain every number, metric, and analysis you need:

1. `paper_reporting/pretraining_report.md` — Full pre-training report (architecture, training, routing analysis)
2. `paper_reporting/base_pretrained__performance.md` — Base model evaluation (10 benchmarks + domain perplexity)
3. `paper_reporting/sft__performance.md` — SFT evaluation (10 benchmarks, forgetting analysis, routing stability)
4. `paper_reporting/training_metrics.csv` — Pre-training step-level metrics
5. `paper_reporting/sft_training_metrics.csv` — SFT step-level metrics
6. `paper_reporting/final_dynamic_entropy.json` — Per-layer routing entropy data
7. `paper_reporting/entropy_stats.json` (in figures/) — Entropy statistics

Also read for context:
- `CLAUDE.md` — Project architecture and specs
- `src/model/architecture.py` — The actual PolyGLU implementation (frozen code)
- `model_card_README.md` — Base model HuggingFace card
- `model_card_instruct_README.md` — Instruct model HuggingFace card

**CRITICAL**: Every number in the paper MUST come from these source files. Do NOT invent metrics. If a number isn't in the source materials, don't include it.

---

## Step 2: Paper Narrative and Structure

### Core Thesis

PolyGLU introduces **state-conditional activation routing** into the transformer FFN, inspired by neurotransmitter-receptor diversity in biological neural networks. The key finding is NOT benchmark scores (the model is small and budget-constrained) — it is the **emergent routing behavior**:

1. **Routing converges to near-deterministic selections** (entropy = 0.03% of maximum) WITHOUT any explicit sparsity loss — this is purely emergent
2. **Depth-dependent activation specialization** emerges: early layers prefer GELU, deep layers prefer Tanh — a learned computation gradient
3. **Routing is fully robust to fine-tuning**: entropy stays at maximum (1.386 = ln(4)) throughout 13,067 SFT steps — the architecture cleanly separates "how to compute" from "what to compute"
4. **All of this with only 0.23% parameter overhead** (~1.4M routing params out of 597M)

### Narrative Arc

The paper should follow this progression:

1. **Motivation**: Biological neural systems use diverse neurotransmitters (glutamate, GABA, dopamine, acetylcholine) that determine how signals are processed. Current transformers use a single fixed activation. What if we let neurons choose?

2. **Method**: PolyGLU — each neuron has a static preference (α) and dynamic gating (β·f(h)) over K=4 activations, combined via Gumbel-Softmax. Minimal overhead (0.23%), drop-in SwiGLU replacement.

3. **Emergent Behavior** (THE core contribution):
   - Near-deterministic routing emerges without regularization
   - Layer-wise specialization: GELU→mixed→Tanh gradient across depth
   - Three "polyglot" layers (9, 16, 17) maintain elevated entropy — potential computational flexibility points
   - The neurotransmitter heatmap (figures/neurotransmitter_heatmap.png) is a visually striking figure that should be prominently featured

4. **Validation at Scale**: 597M params, 10B tokens, single A100 — achieves 62-89% of Qwen3-0.6B-Base (trained on 3,600x more data) on standard benchmarks

5. **Fine-tuning Robustness**: SFT on math data shows routing entropy is completely stable (1.386 constant) — the routing architecture is a permanent structural feature, not a fragile training artifact. This is a key result for practical deployment.

6. **Honest Limitations**: Budget-constrained research (~$346 total), no GSM8K due to lack of KV cache, no ablation against vanilla SwiGLU baseline. Frame these as future work, not apologies.

### Suggested Structure

```
1. Introduction
2. Related Work (activation functions, MoE, Gumbel-Softmax, adaptive computation)
3. Method
   3.1 PolyGLU Formulation
   3.2 Routing Mechanism (static + dynamic)
   3.3 Gumbel-Softmax Temperature Annealing
   3.4 Integration into Transformer Block
4. Experimental Setup
   4.1 Model Architecture (597M params, specs table)
   4.2 Pre-training (10B tokens, data mix, hardware)
   4.3 Supervised Fine-tuning (Nemotron-Math-v2, ChatML)
   4.4 Evaluation Protocol (lm-eval-harness)
5. Results
   5.1 Pre-training Convergence
   5.2 Emergent Routing Behavior (THE key section — dedicate space)
      - Near-deterministic convergence
      - Layer-wise specialization
      - Neurotransmitter maps
   5.3 Benchmark Performance (base model)
   5.4 Domain Perplexity
   5.5 Fine-tuning Stability
      - Routing entropy preserved at maximum
      - Forgetting analysis (moderate, acceptable)
      - MMLU-STEM improvement as transfer evidence
6. Analysis and Discussion
   6.1 Why Does Routing Converge Without Regularization?
   6.2 The Static-Dynamic Separation
   6.3 Implications for Activation Function Design
   6.4 Limitations and Future Work
7. Conclusion
```

---

## Step 3: Figures to Use

These are available in `paper/figures/` (copied from `paper_reporting/figures/`). Use them generously — they are a strength of this work:

### Must-include (core results):
- `neurotransmitter_heatmap.png` — THE signature figure. Shows preferred activation per neuron across all 28 layers. Visually striking, reveals layer specialization.
- `layer_distribution.png` — Activation function preference distribution by layer (GELU→Tanh gradient)
- `dynamic_routing_entropy_final.png` — Per-layer routing entropy at convergence (near-zero except layers 9, 16, 17)
- `loss_curve.png` — Pre-training loss (12.13 → 1.31)
- `eval_benchmark_comparison.png` — Base model vs Qwen3-0.6B-Base benchmarks
- `sft_base_vs_sft_benchmarks.png` — Base vs SFT benchmark comparison with Qwen3 reference
- `sft_delta_chart.png` — Per-benchmark SFT impact

### Strongly recommended:
- `combined_training_dynamics.png` — Loss, LR, tau, throughput in one figure
- `tau_annealing.png` — Gumbel-Softmax temperature schedule
- `eval_domain_perplexity.png` — Math/Code/STEM perplexity
- `sft_loss_curve.png` — SFT training loss
- `dynamic_entropy_evolution.png` — Entropy evolution during training
- `entropy_histogram.png` — Distribution of per-neuron entropy values

### Optional:
- `throughput.png`, `lr_schedule.png` — Training infrastructure details
- `eval_all_benchmarks.png`, `eval_category_summary.png` — Alternative benchmark views
- `sft_training_dynamics.png` — SFT training dynamics
- `routing_shift_summary.png`, `dynamic_routing_comparison.png` — Routing analysis extras

---

## Step 4: References — CRITICAL ANTI-HALLUCINATION PROTOCOL

**This is the most important instruction in this entire prompt.**

LLMs frequently hallucinate paper titles, authors, years, and venues when generating BibTeX entries. This is unacceptable for an arXiv submission. Follow this protocol strictly:

### Protocol:
1. **NEVER generate a reference from internal knowledge.** Every single BibTeX entry must be verified via web search.
2. For each reference you need, use the WebSearch tool or spawn a subagent (Agent tool, subagent_type="general-purpose") to:
   - Search for the exact paper title + authors
   - Verify the year, venue/journal, and arXiv ID
   - Confirm the paper actually exists
3. **After writing all references**, spawn a dedicated verification subagent that checks EVERY entry in `references.bib`:
   - Search for each paper by title
   - Verify authors, year, venue match
   - Flag any entry that cannot be confirmed
   - Remove or fix any unverifiable entries
4. Prefer arXiv IDs (e.g., `2305.14314`) as they are easily verifiable
5. When in doubt, cite fewer papers rather than risk a fake citation
6. Do NOT cite papers just because they seem like they should exist — only cite papers you have VERIFIED exist

### Papers you will likely need to cite (VERIFY ALL OF THESE):
- Original Transformer (Vaswani et al., "Attention Is All You Need")
- SwiGLU / GLU variants (Shazeer, 2020; Dauphin et al., 2017)
- Gumbel-Softmax (Jang et al., 2017; Maddison et al., 2017)
- RoPE (Su et al., 2021)
- RMSNorm (Zhang & Sennrich, 2019)
- GQA (Ainslie et al., 2023)
- Mixture of Experts (Shazeer et al., 2017; Fedus et al., 2022)
- Flash Attention (Dao et al., 2022)
- DeepSpeed (Rasley et al., 2020)
- lm-evaluation-harness (Gao et al.)
- Qwen3 technical report
- AdamW (Loshchilov & Hutter)
- Relevant biology/neuroscience references for the neurotransmitter analogy (BE VERY CAREFUL — verify these exist)

---

## Step 5: Specific Content Instructions

### Abstract (~200 words)
Lead with the biological inspiration, state the method (PolyGLU), highlight the emergent routing behavior as the key finding, mention scale (597M, 10B tokens, single A100), note benchmark context (62-89% of Qwen3 at 3,600x less data), and the fine-tuning stability result.

### Introduction
- Open with the observation that biological neural systems use diverse neurotransmitters
- Contrast with the "one activation fits all" paradigm in current transformers
- State the contribution: PolyGLU, emergent routing, fine-tuning robustness
- Note this is independent research on a single GPU — frame budget constraints as a feature (reproducibility, accessibility) not a bug

### Links to include prominently
- GitHub: https://github.com/danielxmed/PolyGLU
- Base model: https://huggingface.co/tylerxdurden/PolyChromaticLM-1.0-base-0.6B
- Instruct model: https://huggingface.co/tylerxdurden/PolyChromaticLM-1.0-instruct-0.6B

### Things to explicitly mention
- Optimized inference code with KV cache and complete GSM8K evaluation results will be released separately in a follow-up publication
- The model, weights, and all training code are fully open-source under Apache 2.0
- All experiments were conducted on a single NVIDIA A100 80GB GPU rented from RunPod community cloud at ~$1.64/hr, with a total project budget of approximately $346

### Tone
- Confident but honest. This is a small model with a big idea.
- Don't oversell benchmark numbers — they're respectable for the compute budget, not state-of-the-art
- DO sell the emergent routing behavior — it's genuinely novel and interesting
- The neuroscience analogy should be presented as inspiration, not as a claim of biological equivalence
- Frame limitations as opportunities for the community to build upon

---

## Step 6: LaTeX Specifics

- Use `\documentclass{article}` with standard arXiv-compatible packages
- Use `natbib` for citations
- Use `graphicx` for figures, `booktabs` for tables
- Keep it to ~8-10 pages (main text) + references + appendix if needed
- Author: Daniel Nobrega (no affiliation — independent researcher)
- Include `\usepackage{hyperref}` for clickable links

---

## Step 7: Final Verification Checklist

Before declaring the paper complete:

1. [ ] Every number matches the source reports (cross-check key metrics)
2. [ ] All figures referenced in text actually exist in `paper/figures/`
3. [ ] All BibTeX entries have been web-verified (run verification subagent)
4. [ ] No hallucinated references remain
5. [ ] Links to GitHub and HuggingFace repos are correct and included
6. [ ] Paper compiles with `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
7. [ ] Future work mentions KV cache + GSM8K follow-up publication
8. [ ] Limitations section is honest about budget constraints and missing evaluations
