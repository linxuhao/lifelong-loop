# The Lifelong Loop

Code, raw per-night/per-fact timelines, and paper for:

> **The Lifelong Loop: Error-Gated Consolidation and the Cost of Nights Without a Log** (v1 preprint, July 2026 — [paper/lifelong_loop_v1.pdf](paper/lifelong_loop_v1.pdf))

Companion to *Persistence Is Not Accumulation* (doi:10.5281/zenodo.21232648, code doi:10.5281/zenodo.21199026),
which audits the **day** scale; this paper runs the complete **day/night loop** end-to-end for up to
30 simulated days (720 facts) on a frozen Qwen3.5-2B with a disposable LoRA day adapter and a
persistent LoRA core:

1. **Error-gated consolidation** (train the core only on what it currently fails, from the log) retains
   **2.2×** recency-only consolidation at identical nightly budget, by preferentially rescuing the
   *latent* class (recognized-but-not-recalled: 0% next-day survival under recency-only, 46% under
   error-gating).
2. **Nights without a log poison the system**: pure self-recitation collapses structurally (0/144), and
   mixing recitation with ground truth lands *below* doing nothing about the past (4.3 vs 25.0/144) —
   fluent wrong recitations overwrite earlier consolidation (error compounding).
3. **No exhaustion by 30 days / 720 facts**: full replay holds 99% (r32 core streaming capacity exceeds
   the generator); the error-gated arm's retained fraction *rises* with horizon (38→44→46%), settling
   into a throttled steady state (old-day plateau, squeezed intake of the newest days).
4. **The served agent is not aphasic** (GSM8K intact in both serving states) but the hot-day mount
   carries a ~6× perplexity accent on free text and masks old-fact recall 3–10× at read time —
   prescription: consolidate nightly, serve from the core, mount the day adapter narrowly.

Capability firewall (adapters-off GSM8K == base, +0.00) in every run.

## Layout
```
loop.py               # the end-to-end day/night loop (one driver, all night arms + dual-state probes)
lib.py                # probes, GSM8K firewall eval (shared with the companion instrument)
gsm8k_pilot_ids.json  # frozen 10-item GSM8K firewall subset
analyze.py            # re-derives every table from results/ (pinned definitions in its docstring)
results/              # raw per-night, per-fact timelines for every run in the paper
paper/                # LaTeX source, figure, PDF
reproduce.sh          # full run matrix
```

## Reproduce
```bash
pip install torch transformers peft datasets
python analyze.py     # re-derive all paper tables from shipped raw results
bash reproduce.sh     # re-run everything (fp32 Qwen3.5-2B; ~10GB VRAM; D30 runs take hours each)
```

## Honest notes
- One substrate at loop level; synthetic collision-free facts; 24 facts/day; one nightly budget.
  1–3 seeds per condition on nondeterministic consumer ROCm — directions, not points; per-seed
  numbers always shown.
- `*_oncap` result files are re-runs of the D6 error-gated arm with dual-state capability probes
  (training identical; filenames encode the flag). Trained core checkpoints (~135MB each) exceed
  GitHub file limits and are archived separately; available on request / with the Zenodo record.
- The misslog self-test cost grows linearly with history (generation, not gradients); a deployment
  would subsample it — untested here.

MIT license (see LICENSE).
