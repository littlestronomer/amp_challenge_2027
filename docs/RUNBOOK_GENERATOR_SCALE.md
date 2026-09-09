# Frozen generator validation at scale

No further training or production promotion. Compare the original primary
generator, all three RAFT endpoints, and all three GRPO endpoints. The baseline
is sampled with the matching fresh sampling seed for each endpoint pair: nine
cells total, not nine independently trained models. This does not reproduce the
two-generator production hybrid.

## Fixed protocol

- Inputs: all six completed `*-generator-seed{42,43,44}-v1` pilots. Source hashes,
  frozen predictor weights, calibration, shared backbone revision, initial model,
  reference and source code must match. No best-seed choice.
- Fresh sampling seeds: 40042/40043/40044, distinct from the pilot evaluation
  seeds 10042/10043/10044. Full masked categorical sampler, temperature1, lengths
  8..50, batches32, default checkpoint conditioning; no top-p/repetition changes.
- Default budget: exactly100,000 raw draws per cell. No filtering during sampling,
  no policy updates, no automatic budget expansion. Pool prefixes2048/8192/32768/
  100000 are paired nested observations, not independent replicates.
- Libraries: first50,000 distinct non-reference sequences in draw order, with no
  activity/risk filtering. Record draws to reach50k, duplicates and reference
  matches. If insufficient, save the raw pool and shortfall; do not fabricate a
  library or quietly resample. Duplicate/reference counts may overlap.
- Frozen reward and evaluation heads are exactly those pinned in the pilots.
  Evaluation shares backbone/data lineage and is not independent biological
  validation. Fresh sampling does not create a fresh labeled test set.

## Execute on SSH

Run from the repository. These commands need existing ML and seqme dependencies;
no dependency changes were introduced.

```bash
git pull --ff-only origin main

uv run --no-sync python scripts/validate_generator_scale.py sample \
  --out sweep_results/generator-scale-v1 --list

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/validate_generator_scale.py sample \
  --out sweep_results/generator-scale-v1
```

The sample command runs all nine cells sequentially:900,000 total raw draws plus
35M scoring. It is substantially larger than the pilots. It never loads650M or
15B during sampling. Baseline/RAFT/GRPO use identical draw budgets. Training cost
has already been incurred and is not erased by this generation comparison.

Optionally split work into cells using `--method baseline|raft|grpo --seed 42|43|44`.
All commands must use the same device/software and draw budget. Do not run writers
concurrently against the same output tree. Repeating sample verifies and skips
completed cells. An interrupted **sampling cell** is preserved and rejected:
mid-cell RNG/optimizer recovery is not implemented. Use a new output root if that
happens; do not delete or overwrite the failed cell. GPU models are released
between sampling cells, and all parameter gradients are disabled.

CPU audit (no new generation/scoring):

```bash
uv run --no-sync python -u scripts/validate_generator_scale.py audit \
  --out sweep_results/generator-scale-v1

uv run --no-sync python scripts/validate_generator_scale.py report \
  --out sweep_results/generator-scale-v1
```

The exact greedy separation and reference comparisons can be CPU intensive at
100k. Progress is printed per scale and every250 shortlisted reference checks.
The CPU audit and650M evaluation can retry incomplete calculations under the same
stage recipe; completed stage hashes are checked before reuse.

Inspect library shortfalls and selector support before expensive evaluation:

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("sweep_results/generator-scale-v1/report")
print(pd.read_csv(root / "status.csv").to_string(index=False))
print((root / "report.json").read_text())
if (root / "frontier.csv").exists():
    print(pd.read_csv(root / "frontier.csv").to_string(index=False))
PY
```

For complete50k libraries, run the existing strict component-metric suite with
the fixed650M embedder. Each evaluation runs in its own subprocess:

The new runner sets Python/NumPy/Torch RNG seed2027 for every official evaluation;
this controls seeded randomness but does not guarantee cross-platform metric
identity. Existing evaluator callers retain their behavior unless they pass the
new optional `--seed` argument.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/validate_generator_scale.py evaluate \
  --out sweep_results/generator-scale-v1

uv run --no-sync python scripts/validate_generator_scale.py report \
  --out sweep_results/generator-scale-v1
```

Incomplete libraries are explicitly skipped. A failed evaluator reports its log
tail; successfully evaluated cells are reused on retry. Reports are regenerated
from completed, hash-verified cells. Original samples and audits are immutable.
Send `report/status.csv`, `report/growth.csv`, `report/frontier.csv`,
`report/official.csv` when present, and `report/report.json`.

## What the audit measures

`growth.csv`: at every raw prefix, unique qualifying yield per all draws, unique
non-reference fraction, mean/p75 evaluation risk, raw length, first256 pairwise
distance, and score-ordered greedy similarity-separated qualifying count.
Qualification uses activity>=.6 and both risks<=.5. Separation rejects similarity
>=.8. The greedy result is not a full family graph or maximum independent set;
counts are not expected to grow monotonically when new high-ranked sequences
change greedy choices. The first256 diagnostic is deliberately identical across
prefix sizes, not fresh diversity evidence at each scale.

`lengths.csv`: raw shares, activity/risk and qualifying yield within8–14/15–24/
25–34/35–50 bins. Empty bins have null rates, not zeros. Length stratification is
descriptive, not full control of sequence composition or predictor bias.

`frontier.csv`: select from each complete50k library, not the entire100k pool.
First take up to2000 unique candidates with activity>=.6, ordered only by training
activity minus training risk. Check every shortlisted candidate against all
references, rejecting similarity **>.8** (equality passes), matching the repo's
submission verifier. Safe length bounds prune impossible violations; no
approximate nearest-neighbor search is used. Do not replace failing candidates
from outside this fixed shortlist.

For each mean-distance floor .50/.55/.60/.65, select greedily in that score order
with pairwise similarity<.8 and cumulative mean distance>=the floor. Return up to
100; never relax constraints. The evaluation head is used only to report the
result. The same floor does not guarantee equal achieved diversity. The report
marks a seed/target as matched only if all three methods reach100 and their
achieved mean distances span at most .01. The .01 tolerance is exploratory, not
a significance threshold. Inspect the frontier rather than choosing a target
post hoc from evaluation-risk scores. Shortlists and greedy choices can cause
shortfalls even when a different algorithm could find a feasible set.

`audit/selected.csv`: actual sequences/ranks and all scores for every floor.
These are diagnostic selections, not promoted `top.fasta` submissions. The
library passes exact-reference exclusion and structural checks; selected
candidates additionally pass the repo's reference-similarity rule. This tool
does not clone/install/re-execute the production entry point, so it is not the
full submission reproducibility validator.

Official metrics describe the50k library, not the top100 frontier. They remain
component scores, not the hidden competition ranking. Strong predictor gains do
not justify a safety claim, and training thresholds must not be tuned against
these evaluation results without declaring a new development iteration.
