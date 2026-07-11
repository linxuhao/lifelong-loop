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
3. **The ordering is an architecture fact**: all five night arms replicate structurally on
   SmolLM2-1.7B-Instruct (hyperparameters unchanged) — error-gated 2.0× recency (vs 2.2× on Qwen),
   recitation collapse, hybrid poisoning (even harder), full-replay ceiling 94%.
4. **No exhaustion by 30 days / 720 facts**: full replay holds 99% (r32 core streaming capacity exceeds
   the generator); the error-gated arm's retained fraction *rises* with horizon (38→44→46%), settling
   into a throttled steady state (old-day plateau, squeezed intake of the newest days).
5. **The served agent is not aphasic** (GSM8K intact in both serving states) but the hot-day mount
   carries a ~6× perplexity accent on free text and masks old-fact recall 3–10× at read time —
   prescription: consolidate nightly, serve from the core, mount the day adapter narrowly.

Capability firewall (adapters-off GSM8K == base, +0.00) in every run.

## Layout
```
loop.py               # the end-to-end day/night loop (one driver, all night arms + dual-state probes)
lib.py                # probes, GSM8K firewall eval (shared with the companion instrument)
gsm8k_pilot_ids.json  # frozen 10-item GSM8K firewall subset
analyze.py            # re-derives every table from results/ (pinned definitions in its docstring)
make_figure.py        # regenerates the paper figure from results/
results/              # raw per-night, per-fact timelines for every run in the paper
paper/                # LaTeX source, figure, PDF
reproduce.sh          # full run matrix
```

## Reproduce
```bash
pip install -r requirements.txt
python analyze.py     # re-derive all paper tables from shipped raw results
python make_figure.py # regenerate the paper figure
bash reproduce.sh     # re-run everything (fp32 Qwen3.5-2B; ~10GB VRAM; D30 runs take hours each)
```

Single runs:

```bash
python loop.py --night misslog --days 6 --firewall-n 10 --seed 1234    # the headline arm
python loop.py --night markov  --days 6 --firewall-n 10 --seed 1234    # recency baseline
python loop.py --night hybrid  --days 6 --firewall-n 10 --seed 1234    # the poisoning arm
python loop.py --night misslog --days 30 --probe-stride 3 --seed 1234  # the marathon
python loop.py --night misslog --days 6 --oncap 10 --save-core --seed 1234  # + dual-state capability probes
```

### Driver flags

| flag | values (default first) | meaning |
|---|---|---|
| `--night` | markov · full · misslog · recite · hybrid | nightly consolidation policy (see paper §2) |
| `--days` / `--fpd` | 6 / 24 | horizon in days / facts per day |
| `--probe-stride` | 1 | full-history post-night probe every k nights (mechanism self-tests unaffected) |
| `--oncap` | 0 | >0: GSM8K items for mounted-state capability probes (+ fixed-text NLL) |
| `--save-core` | off | archive the trained core adapter to `results/cores/` |
| `--core-epochs` | 3 | nightly gradient budget = fpd × core_epochs steps (all arms except `full`) |
| `--model` | Qwen/Qwen3.5-2B | substrate (non-default encoded in the filename) |
| `--core-rank --day-rank --ws --replay-m --ewc-lambda --lr --seed` | 32 · 64 · 8 · 4 · 300 · 3e-5 · 1234 | adapters / day-phase / opt |

### Result JSON schema

Filenames: `loop_{night}[_oncap][_M{model}]_D{days}_f{fpd}_s{seed}.json`. Fields:

- top level: hyperparameters, `firewall` (`{gsm8k_base, gsm8k_off}`, plus `ppl_base` when `--oncap`);
- `trajectory`: one entry per night. Full-probe nights have `post_hits` (per-fact 0/1 core-only
  recall over ALL facts so far; index = fid, day of write = fid // fpd), `margins` (2-AFC logprob
  margin per fact, >0 = recognized), `recall_by_day` (post_hits summed per write-day — the age
  curve), `n_recognized`, `pre_night_hits` (core+day recall before consolidation — the read-masking
  comparison), and with `--oncap` an `oncap` dict (`gsm8k_core_day`, `ppl_core_day` = hot daytime
  mount; `gsm8k_core`, `ppl_core` = morning core-only). Stride-skipped nights have only
  `pre_night_hits` and `today_post` (today's cohort pulse).

`analyze.py` legend: `final` = post-night recall of the whole history at the last night ·
`recog` = final 2-AFC counts · `survivalRLG` = next-night survival of facts that were **R**ecalled /
**L**atent (recognized-only) / **G**one at the previous probe — the category table of paper §3.1.

**Hardware note**: the code is standard PyTorch + transformers + peft — it runs on NVIDIA/CUDA
as-is (`--dev cuda:0` is the same device string under ROCm, which is simply what *our* GPUs were).
The ROCm remarks in this repo describe our measurement hardware, not a requirement; expect exact
numbers to differ on any hardware (they differ across our own seeds too).

## Honest notes
- The D6 policy comparison is two-substrate; D12/D30 horizons are Qwen-only. Synthetic collision-free
  facts; 24 facts/day; one nightly budget.
  1–3 seeds per condition on nondeterministic consumer ROCm — directions, not points; per-seed
  numbers always shown.
- `*_oncap` result files are re-runs of the D6 error-gated arm with dual-state capability probes
  (training identical; filenames encode the flag). Trained core checkpoints (~135MB each) exceed
  GitHub file limits and are archived separately; available on request / with the Zenodo record.
- The misslog self-test cost grows linearly with history (generation, not gradients); a deployment
  would subsample it — untested here.

MIT license (see LICENSE).
