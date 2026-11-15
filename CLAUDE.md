# CLAUDE.md: mi-genderbias-mistral-steer

## What this repo is
A small, learning-oriented mechanistic interpretability (MI) demo for an independent research paper. It finds, explains and removes **occupational gender bias** in **Mistral-7B-Instruct-v0.3**. The tools are deliberately foundational: logit lens, activation patching, a **linear probe**, and a **difference-of-means direction** (Contrastive Activation Addition, Rimsky et al. 2024) used for steering.
The sister repo `olawoye/mi-genderbias-qwen-nla` runs the same pipeline on Qwen2.5-7B, using Anthropic's Natural Language Autoencoders for the explain step. Keep the shared files (`prompts.py`, `checkpoint.py`, `metrics.py`, `hooks.py`, `locate.py`, `steer.py`, `data/`) **identical** across both repos.

## Pipeline (one notebook per step)
00 setup check → 01 baseline bias (she−he logit diff, BLS correlation, résumé A/B task) → 02 locate (logit lens, activation patching per layer) → 03 explain (per-layer linear probe for gender; diff-of-means direction; cosine of each occupation's activation with that direction vs BLS %-female) → 04 steer + controls (projection ablation / activation addition, random-direction control, perplexity).

## Rules
- Use plain PyTorch forward hooks on HF `transformers` models, not TransformerLens or nnsight. Writing the hooks by hand is the point.
- All logic lives in `src/mibias/`; notebooks only call functions and plot.
- Mistral uses a SentencePiece tokenizer: get target token ids **in context** (prefix vs prefix+target), never by encoding `" she"` alone.
- Build the gender direction only from `prompts.contrast_pairs()`. These contain no occupation words, which keeps the evaluation non-circular.
- Every result is written to `cfg.results_dir` (Drive on Colab) as JSON + PNG, with seed and model id recorded; `checkpoint.export_results` copies them into `results/` for committing.
- Every expensive step goes through `checkpoint.open_run(cfg)`: `cached()` for one-shot steps, `map_chunks()` for loops over prompts. Return numpy, not torch. Never write results with plain `open()`.
- Local machine is CPU-only: `MIBIAS_SMOKE=1` swaps in a tiny random Mistral (tests the plumbing only). `pytest -q` must pass without downloads.
- Runs on a Colab L4 (24 GB) in bf16.

## Known TODOs
- `data/occupations.csv` %-female values are **approximate**. Verify them against the BLS CPS Table 11 (latest year) before quoting in the paper.
