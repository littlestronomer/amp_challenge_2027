# N-way library blend + multi-axis conditioned secondaries

This iteration generalizes the locked two-pool hybrid (75/25, interior FBD/MMD
minimum) to N components, and adds new components conditioned on the full
ConformityScore property space (charge + hydrophobicity + hydrophobic moment,
jointly distributed). No checkpoint, selector default, or submission artifact
changes automatically; every promotion below is a separate reviewed step.

Motivation and prior evidence: blending is the only mechanism with a proven
interior optimum at the 650M gate (FBD 0.269 → 0.221 at 75/25); Conformity
behaves mixture-linearly; the epoch-58 pools measured FBD 0.211 / MMD 0.233 as
whole libraries but failed as a *replacement* — both suggest multi-component
mixtures as the cheapest remaining lever. The incumbent (650M): FBD 0.221,
MMD 0.357, Recall 0.899, Precision 0.863, Conformity 0.507, Diversity 0.848.

## Stage 0 — pending reports + component inventory (read-only, CPU)

Run first; the pending post-training numbers decide whether GRPO/RAFT/OPD
endpoints join the component pool.

```bash
# Pending post-training reports (send back the printed tables)
uv run --no-sync python - <<'PY'
import pandas as pd
from pathlib import Path
for name, paths in {
    "pool-accounting": ["sweep_results/pool-accounting-v1/pool_accounting.csv"],
    "opd-evaluation": ["sweep_results/opd-evaluation-v1/report/status.csv",
                        "sweep_results/opd-evaluation-v1/report/paired_growth_summary.csv",
                        "sweep_results/opd-evaluation-v1/report/frontier.csv"],
}.items():
    for p in paths:
        if Path(p).exists():
            print(f"\n## {p}")
            print(pd.read_csv(p).to_string(index=False))
        else:
            print(f"\n## MISSING {p}")
PY

ls sweep_results/grpo-pilot-comparison-v2 2>/dev/null || echo "no grpo comparison dir"
find sweep_results -maxdepth 1 -type d -name '*scale*' -o -maxdepth 1 -type d -name '*external-pilot-v2*' | sort

# Component pool inventory: sizes + hashes of candidate pools
find sweep_results generate -name 'library.fasta' -o -name 'pool.fasta' | sort | \
  xargs -I{} sh -c 'printf "%8d  %s  {}\n" "$(grep -c "^>" {})" "$(sha256sum {} | cut -c1-12)"'
```

Component selection rule (pre-declared): pick 3–5 pools spanning the metric
trade-off axes — (a) the incumbent primary/cond pools, (b) one
typicality/precision-leaner pool (epoch-58 blends or a v2-style e100_p10
seed), (c) at most one post-training endpoint pool, and only if its pending
report shows qualifying-yield gains without diversity/reference-reproduction
collapse. Every component must be a clean pool (valid, unique,
reference-free) — `scripts/sweep_nblend.py` verifies this and fails otherwise.

## Stage 1 — N-way ratio probe (no training)

Regenerate the incumbent hybrid control into a fresh directory (byte-stable),
then run the probe with the Stage-0 component pools:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --extra ml generate \
  --out-dir generate/incumbent-hybrid-nblend-v1

uv run --no-sync python scripts/sweep_nblend.py \
  --component primary=<stage-0 primary pool.fasta> \
  --component cond=<stage-0 charge-cond pool.fasta> \
  --component ep58=<stage-0 epoch-58 pool.fasta> \
  --control incumbent=generate/incumbent-hybrid-nblend-v1/library.fasta \
  --ratios 9:3:1 6:2:1 5:2:1 11:4:1 19:6:1 \
  --out sweep_results/nblend-probe-v1 --list
```

`--list` validates pools, quotas and feasibility without writing. Then build
and triage (8M embedder), and gate the strongest cells at official fidelity
(650M) in the same run:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/sweep_nblend.py \
  --component ... --control ... --ratios 9:3:1 6:2:1 5:2:1 11:4:1 19:6:1 \
  --gate --gate-top 4 \
  --out sweep_results/nblend-probe-v1
```

Notes:

- Ratio weights are positive integers; component order = flag order. Quotas
  use exact Hamilton apportionment, so ratios need not divide the library
  size.
- Triage gate-in rule: cells with 8M FBD above control FBD + 0.05 are not
  gated (the binding PHASE1 lesson: 8M deltas below ~0.05 FBD are not
  decision-grade — triage only eliminates clearly worse cells).
- Interrupted runs resume verified cells; any input change requires a NEW
  `--out` root. Never edit files inside an output root.
- Outputs: per-cell `library.fasta`/`membership.csv`/`blend_stats.json`,
  `results.csv` (triage), `gate_results.csv`, `gate_verdicts.csv`,
  `report.json` (paired deltas vs the in-run control).

Pre-declared decision rule: a cell is a promotion candidate only if, at the
650M gate, FBD AND MMD beat the control and Conformity/Precision/Recall/
Diversity each stay within −0.01. `gate_verdicts.csv` computes exactly these
flags (`within_tolerance`). A negative result is recorded, not re-swept.

## Stage 2 — modlamp parity gate (before any conditioned training)

The local descriptors already reproduce the recorded seqme reference
statistics (hydro −0.0672 ± 0.3808, hmoment +0.3953 ± 0.1987, charge
+2.8464 ± 3.2836 on `data/antibacterial.fasta`) and the default bin edges
clamp ≤1.4% of the reference into edge bins. The SSH run is the authoritative
check against modlamp itself:

```bash
uv run --no-sync python scripts/parity_modlamp_descriptors.py \
  --sample 400 --out experiments/modlamp_parity_v1.json
```

Requirements (gate for Stage 3): exit code 0 (all axes max-abs-diff ≤ 1e-6
vs modlamp) and every reference edge-clamp fraction ≤ ~2%. If parity fails,
stop and send back the JSON — do not adjust constants silently.

## Stage 3 — targeted conditioned SFT runs (2–3 runs)

Train multi-axis secondaries with the confirmed e100/p10 recipe on the
canonical corpus, matching the G1 charge-conditioned invocation except for
`--conditioning`:

```bash
for axes in "charge,hydro" "charge,hydro,hmoment"; do
  CUDA_VISIBLE_DEVICES=1 uv run --extra ml python -u scripts/train_generator.py sft \
    --data data/processed/generative.csv \
    --epochs 100 --patience 10 \
    --residual block_attnres --ffn dense \
    --conditioning "$axes" \
    --seed 42 \
    --out-dir "checkpoint/generator-e100_p10-cond-$(echo "$axes" | tr ',' '-')" || break
done
```

Verify: `config.json` records the canonical conditioning string and the bin
parameters; training logs print per-axis bin histograms.

Then sample a component pool per checkpoint (joint bins auto-drawn from the
reference at generation time):

```bash
CUDA_VISIBLE_DEVICES=1 uv run --extra ml python - <<'PY'
from pathlib import Path
from amp_challenge_2027.data import read_reference_set, write_fasta
from amp_challenge_2027.generate import generate_with_model
from amp_challenge_2027.pipeline import clean_candidates

reference = read_reference_set("data/antibacterial.fasta")
for name in ["charge-hydro", "charge-hydro-hmoment"]:
    ckpt = Path(f"checkpoint/generator-e100_p10-cond-{name}")
    out = Path(f"sweep_results/nblend-cond-pools-v1/{name}")
    out.mkdir(parents=True, exist_ok=True)
    raw = generate_with_model(
        160000, seed=42, length=50, device="cuda", checkpoint_dir=ckpt,
        reference_set=reference,
    )
    pool = clean_candidates(raw, reference)
    print(f"[cond-pool] {name}: {len(raw)} raw -> {len(pool)} clean")
    write_fasta(pool, out / "pool.fasta")
PY
```

Standalone 650M gate per new generator (G1 precedent says standalone coverage
may fail — that is acceptable for a blend component), then re-run the Stage-1
probe with the new pools as components at small fractions (e.g. 11:3:1:1,
9:3:1:1) in a NEW output root (`nblend-probe-v2`).

## Stage 4 — decision and (manual) promotion

Review `gate_verdicts.csv` across probes. Any promotion is a separate,
explicit step: wire the winning weight vector into the entry point behind
byte-determinism against the winning probe library, rerun
`scripts/verify_submission.py` on a cold clone, and update
`PHASE1_RESULTS.md`, `docs/ABSTRACT.md` and `docs/DATA_DISCLOSURE.md`
(disclose joint-bin conditioning and component provenance). If no cell clears
the tolerance, record the negative result; the architecture probe and the
top-100/predictor-alignment track are the documented next options.

## Non-goals / future work

No new architecture search, decode sweeps, or selector changes here. The
MoE/depth-width probe, amPEPpy consensus selection, constrained GRPO and the
mixture-inference comparator remain separate future experiments.
