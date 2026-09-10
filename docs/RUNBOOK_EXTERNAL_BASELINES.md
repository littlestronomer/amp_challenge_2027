# External raw-generator benchmark

This experiment compares frozen generators under our local scoring and selection
protocol. It does **not** reproduce native starter-kit submissions or the hidden
competition aggregation score. No training, production promotion, or Kaggle
submission occurs. Existing internal results and checkpoints are read-only.

## Protocol and important differences

Sources inspected September 10, 2026:

- [AMP-Diffusion kit](https://github.com/szczurek-lab/ampdiffusion-starter-kit),
  commit `1a862af9078e6b55c87d1fa576f3da81851ba94b`.
- [HydrAMP kit](https://github.com/szczurek-lab/hydramp-starter-kit),
  commit `7804df862872ccc6d09fe01c41bafbca194cfa31`; its inference dependency is
  pinned to `6590d2f4c2963f25d30669052a4c4a857e0e7279`.

AMP-Diffusion uses the kit's EMA checkpoint, 1000-step sampler, ESM2-8M decoder,
argmax decoding and uniform batch lengths10..40. We stop at a fixed decoded-draw
budget before deduplication or reference filtering. Batch size32 differs from
the kit's default256; the fixed setting is part of this benchmark. Strict Torch
determinism errors are fatal, rather than silently continuing.

`hydramp_raw` deliberately uses **non-default** `filter_out=False`,
`properties=False`, `n_attempts=1`, AMP conditioning and softmax decoding.
The default kit repeatedly samples until its AMP/MIC classifier filters accept
enough peptides; treating that count as raw draws would be unfair. The raw path
returns exactly one decoding per latent draw and preserves order (the properties
path sorts by predicted scores). It can produce invalid/short peptides. Those
remain in the saved raw JSON and in every yield denominator. Valid-only score
means are labeled `valid_*`; invalid peptides never receive invented scores.
HydrAMP's support is at most25 residues, unlike internal8..50 and diffusion10..40.
Inspect `audit/*/seed*/lengths.csv` before interpreting differences.

**Do not describe results as beating the native HydrAMP submission.** Both kits'
native top100 selectors are replaced by our common fixed-shortlist selector;
the diffusion APEX pipeline is not run. This isolates generators and does not
measure complete published pipelines. Raw draws are matched, not compute,
training data, architectures, or access to independent biological validation.

## 1. Install isolated environments on SSH

From the main repository, after `git pull --ff-only origin main`:

```bash
mkdir -p external_baselines

GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/szczurek-lab/ampdiffusion-starter-kit.git external_baselines/ampdiffusion
git -C external_baselines/ampdiffusion checkout --detach 1a862af9078e6b55c87d1fa576f3da81851ba94b
git -C external_baselines/ampdiffusion lfs pull --include="checkpoint/model.pt" --exclude=""
(cd external_baselines/ampdiffusion && uv sync --frozen)

GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/szczurek-lab/hydramp-starter-kit.git external_baselines/hydramp_raw
git -C external_baselines/hydramp_raw checkout --detach 7804df862872ccc6d09fe01c41bafbca194cfa31
(cd external_baselines/hydramp_raw && uv sync --frozen)
```

These commands download third-party code and dependencies; run sequentially and
stop on any error. Git LFS must be installed for diffusion. Only its generator
checkpoint is fetched; APEX weights are not needed. Do not install either kit
into this project's `.venv`. HydrAMP requires Python3.8/TF2.2; diffusion uses
Python3.10/Torch2.5.1 CUDA12.1. `uv` may need to download the matching Python.
If legacy dependency installation fails, send the error rather than editing
upstream pins or substituting newer TensorFlow. No sudo/CUDA-system changes are
required by this runner. HydrAMP runs on CPU to avoid legacy CUDA dependencies.
Diffusion must have a compatible GPU; it never silently falls back to CPU.

No installation or real pretrained-model run was tested on the development
laptop. Offline tests use mocked third-party inference, plus real local
diagnostics. The remote smoke test is required before large generation.

## 2. Preflight and repeatability smoke test

The control source defaults to the completed `sweep_results/opd-evaluation-v1`
run. All15 internal sample cells must still exist. Their scored prefixes are
reused without regeneration. Predictor/reference hashes and scorer source files
must match the cached evaluation. Sampling streams60042/60043/60044 are paired
labels, not identical latent randomness across architectures.

```bash
uv run --no-sync python scripts/benchmark_external.py preflight \
  --out sweep_results/external-smoke-v1 --draws 32

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/benchmark_external.py sample \
  --out sweep_results/external-smoke-v1 --draws 32

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/benchmark_external.py sample \
  --out sweep_results/external-smoke-repeat-v1 --draws 32
```

Compare decoded sequences, not wall-clock time or JSON bytes:

```bash
uv run --no-sync python - <<'PY'
import json
from pathlib import Path
for method in ["ampdiffusion", "hydramp_raw"]:
    for seed in [42, 43, 44]:
        a = Path(f"sweep_results/external-smoke-v1/raw/{method}/seed{seed}/raw.json")
        b = Path(f"sweep_results/external-smoke-repeat-v1/raw/{method}/seed{seed}/raw.json")
        assert json.loads(a.read_text())["sequences"] == json.loads(b.read_text())["sequences"], (method, seed)
        print("PASS", method, seed)
PY
```

Preflight is read-only: no model loading, environment installation or output
creation. Sampling logs go to `raw/<method>/seed<seed>/generation.log`; failures
print the last35 lines. LFS pointer files, modified tracked sources, missing
environments, wrong commits and changed input hashes are rejected. Dynamic ESM
decoder weights and installed package versions are recorded in raw results;
cross-seed runtime/decoder mismatches fail reporting. The first download's
external ESM weight hashes are recorded, not known in advance from this repo.
Legacy TensorFlow determinism is empirical, not guaranteed by a strict backend.

## 3. Pilot comparison

Only after the smoke test passes:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/benchmark_external.py sample \
  --out sweep_results/external-pilot-v1

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/benchmark_external.py score \
  --out sweep_results/external-pilot-v1

uv run --no-sync python -u scripts/benchmark_external.py audit \
  --out sweep_results/external-pilot-v1

uv run --no-sync python scripts/benchmark_external.py report \
  --out sweep_results/external-pilot-v1
```

Default8192 draws × two external methods × three seeds =49,152 new draws.
Diffusion's1000 steps per batch make it much more expensive than autoregressive
generation; do not treat equal draws as equal GPU time. Worker wall times appear
in `report/generation_costs.csv`; model downloads/loading are included. Cached
internal generation costs are not included or claimed to be zero.

`--method ampdiffusion|hydramp_raw --seed 42|43|44` can restrict sample/score
or external audit work. Report requires all cells. Do not run concurrent writers
against a shared output root. Completed raw cells are hash-verified and skipped;
partial raw cells are preserved and rejected. Scoring/audit may retry under the
same recipe. Do not change code, dependency environments or settings mid-run.

Outputs: `report/growth.csv`, `growth_summary.csv`, `paired_growth.csv`,
`frontier.csv`, `matched_frontier.csv`, `status.csv`, `generation_costs.csv`,
and `report.json`. Reported selection deltas require both sets to reach100 and
achieved distance to differ by at most.01. No invented results for shortfalls.
Our frozen predictors score all structurally valid draws; no external native
classifier is used to rank candidates. The evaluation-only head does not select.

## 4. Optional full scale, after pilot review

Use a new root and the existing100k control source:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/benchmark_external.py sample \
  --controls sweep_results/opd-scale-v1 --draws 100000 \
  --out sweep_results/external-scale-v1

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/benchmark_external.py score \
  --out sweep_results/external-scale-v1

uv run --no-sync python -u scripts/benchmark_external.py audit \
  --out sweep_results/external-scale-v1

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/benchmark_external.py evaluate \
  --out sweep_results/external-scale-v1

uv run --no-sync python scripts/benchmark_external.py report \
  --out sweep_results/external-scale-v1
```

This draws600k new external candidates and may evaluate21 libraries with650M
(including newly evaluated internal controls). Libraries are the first50k
valid unique non-reference draws, with no score-based library selection. If a
generator cannot reach50k within budget, it remains a shortfall; no budget
extension or normalization hides it. Only complete libraries receive the local
component-metric evaluation at fixed seed2027. These are **not official rankings**.
Full-scale generation shares the pilot prefix and is not independent replication.
