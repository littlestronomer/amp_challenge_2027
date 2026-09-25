# Conditional generator research on GPU 1

This runbook trains new generators using measured assay conditions, then tests their
fresh outputs. It leaves the submitted generator, blending recipe and production
selector unchanged. A successful run creates research evidence; it does not establish
experimental activity, safety, or competition eligibility.

Run commands from the repository root on the SSH machine. Use a terminal multiplexer
for the full downloads and training. The training and sampling loops below are
sequential: they share one RTX 5090, not several competing GPU processes.

## 1. Pull, check the environment, and choose new output directories

```bash
cd ~/Documents/Project/amp_challenge_2027
git pull --ff-only origin feat/nway-blend-multiaxis-conditioning
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

uv run --no-sync python - <<'PY'
import torch
print("torch:", torch.__version__, "CUDA build:", torch.version.cuda)
assert torch.cuda.is_available()
print("visible GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.ones(32, device="cuda")
print("GPU kernel check:", (x @ x).item())
PY

uv run --no-sync python -m pytest -q \
  tests/test_assay_model.py \
  tests/test_conditional_data.py \
  tests/test_conditional_training.py \
  tests/test_conditional_research.py \
  tests/test_conditional_integration.py \
  tests/test_conditional_reporting.py \
  tests/test_model.py \
  tests/test_conditioning.py \
  tests/test_multiaxis_conditioning.py

research_root="sweep_results/conditional-generator-v1"
snapshot_root="data/raw/conditional-generator-2026-09-25-v1"
marlys_source="data/processed/generative.csv"
```

Physical GPU 1 becomes logical `cuda:0` after `CUDA_VISIBLE_DEVICES=1`; pass
`--device cuda`, not `--device cuda:1`. Use the existing working CUDA environment.
If dependencies need installing, the repository's configured PyTorch index is
CUDA 12.8; `uv sync --extra ml --extra seqme` installs the training and optional
production-selector scoring dependencies. Do not replace working 5090-compatible
PyTorch with an older baseline repository's build.

Change the two output directory variables before starting a different experiment.
Training, sampling and comparison require new directories. Fetch can resume the
**same** recipe, and preparation can verify an identical completed recipe. Neither
operation silently overwrites a changed snapshot. All subsequent blocks assume the
variables above remain defined in the same shell.

## 2. Obtain immutable source snapshots

The default source is the frozen, curated MarLys-derived CSV already on the SSH
machine. The importer can also accept the original MarLys CSV or FASTA; changing it
changes the experiment and requires a new snapshot/output version.

| Source | Location | Import behavior and terms |
| --- | --- | --- |
| MarLys v3 | [Mendeley record](https://data.mendeley.com/datasets/w4hb5grjwb/3) | Supply a local frozen CSV/FASTA using `--marlys`. The fetcher does not silently substitute a new Mendeley corpus. Existing source disclosures identify the competition corpus as CC0; retain the version-specific source record and upstream citations. |
| DBAASP | [REST API](https://dbaasp.org/api?page=rest) | Fetch paginated IDs and full `/peptides/{id}` JSON cards. Keep database usage policy, citation requirements and original publications; public accessibility is not a blanket redistribution grant. |
| DRAMP | [Official downloads](https://dramp.cpu-bioinfor.org/downloads/) | Fetch the General Data `general_amps.xlsx`. The source page states CC BY 4.0. The legacy URL directory is not a reliable current release number; snapshot timestamps and hashes identify the downloaded bytes. |
| Hemolytik 2, optional | [Zenodo record](https://zenodo.org/records/19699377) and [database](https://webs.iiitd.edu.in/raghava/hemolytik2/) | Disabled by default. Source descriptions/license metadata/notices disagree. Opt-in research snapshots retain an unresolved-terms flag; they are not a license-cleared release recipe. |

Start the normal full snapshot:

```bash
uv run --no-sync python scripts/conditional_generator.py fetch \
  --marlys "$marlys_source" \
  --workers 4 \
  --out "$snapshot_root"
```

This downloads full DBAASP cards and the DRAMP spreadsheet, with bounded workers,
retries and resumable immutable artifacts. Preserve `recipe.json`, `manifest.json`
and the per-artifact source records. Interruptions can be resumed by repeating the
exact command. Do not edit already downloaded raw files.

If you have verified full cards and the General Data spreadsheet locally, use this
**instead**, with paths appropriate to that machine:

```bash
uv run --no-sync python scripts/conditional_generator.py fetch \
  --marlys "$marlys_source" \
  --dbaasp-dir /absolute/path/to/full-dbaasp-json-cards \
  --dramp /absolute/path/to/general_amps.xlsx \
  --out "$snapshot_root"
```

`--dbaasp-dir` declares the supplied collection; it does not claim complete coverage
of the live database. Old flattened DBAASP CSV exports are insufficient because
important chemistry and assay context were discarded. Omitting the reuse flags
fetches the corresponding required data from its source.

`--max-records 20` is for a separate smoke-test snapshot only. Such snapshots are
marked incomplete; `prepare` requires explicit `--allow-incomplete`. Do not use those
results as the full experiment.

For optional Hemolytik research import, start a separate snapshot and add
`--include-hemolytik --hemolytik /absolute/path/to/complete-table.csv`. Without a local
table, the fetcher reads the Zenodo record and preserves its table or ZIP archive,
extracting the named complete-data CSV. Ambiguous archives require an explicit local
table. Do not enable this source in the default recipe until
its conflicting source terms are resolved.

## 3. Prepare and inspect actual supervision before training

```bash
uv run --no-sync python scripts/conditional_generator.py prepare \
  --snapshot "$snapshot_root" \
  --seed 2027 --threshold 0.8 \
  --out "$research_root/data"

cat "$research_root/data/support.json"
```

Preparation writes `molecules.jsonl`, `observations.jsonl`, `dataset.json`,
`support.json` and a hash manifest. It preserves measured intervals, inequality
bounds, uncertainty, endpoint, dose, target, publication and chemistry. MIC, HC50,
MHC and percentage lysis are different endpoints. Unknown metadata stays unknown.

Only compatible molecular identities supervise the competition chemistry. Other
canonical 8–50-residue sequences may remain unconditional language-model examples.
The annotation count can therefore be much smaller than the raw source counts.
MarLys already contains overlapping source databases; new measurements do not imply
new sequences.

All observations/chemical variants of a sequence share its sequence-family
partition. The default fractions are train/validation/calibration/test =
60/15/10/15%, with connected families at `Levenshtein.ratio >= 0.8` (normalized indel
similarity). Vocabularies and numerical normalizers are fitted on training examples.
Calibration and test examples do not train the model or select checkpoints.

Inspect the provided request and its training support:

```bash
uv run --no-sync python - "$research_root/data/dataset.json" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
from amp_challenge_2027.assay_conditioning import fit_condition_schema, condition_support
from amp_challenge_2027.conditional_training import qualifies_for_selective

payload = json.loads(Path(sys.argv[1]).read_text())
train = [r for r in payload["examples"] if r["split"] == "train"]
annotated = [r for r in train if r["supervision_eligible"] and r["conditions"]]
request = json.loads(Path("experiments/conditional_selective_request_v1.json").read_text())["conditions"]
print("Train sequences:", len(train))
print("Annotated train sequences:", len(annotated))
print("Selective joint positives:", sum(qualifies_for_selective(r["conditions"]) for r in annotated))
print("Training endpoints:", Counter(c["endpoint"] for r in annotated for c in r["conditions"]))
print("Training RBC species:", Counter(c.get("rbc_species") for r in annotated for c in r["conditions"] if c["endpoint"] == "%lysis"))
if not annotated:
    raise SystemExit("No compatible observed training conditions: inspect observations before training.")
schema = fit_condition_schema(train)
print("Request support warnings:")
print(json.dumps(condition_support(request, schema), indent=2))
PY
```

**Stop here if suitable supervision is absent.** Inspect excluded observations and
raw cards; restore missing metadata from the original study when justified. Do not
make modified peptides unmodified, turn MHC into HC50, or label missing hemolysis as
safe to increase the count.

The supplied request means E. coli MIC at most 16 µM and human-cell lysis at most
10% at 64 µM. These are experimental request targets, not clinically established
safety thresholds or guaranteed output properties. Unseen categorical values and
out-of-range numeric requests are extrapolations; retain the warnings and distinguish
them from requests supported by training observations. Category spelling matters.
The request's combined biological targets may be unsupported even if each field
appears separately somewhere in training.

The selective control requires at least 32 sequences with suitable measured MIC and
human-cell lysis evidence linked to the same publication. Too few produce
`insufficient_joint_labels`, exit code 2, and **no trained checkpoint**. This is a
legitimate result; it does not stop the conditional arm from using partial labels.

## 4. Run the warm-start controls and conditional fine-tuning

The primary incumbent is `checkpoint/generator`. This experiment tests that
component directly; it does not reproduce the deployed 3:1 blended generator.
The frozen control keeps its weights unchanged. For each trainable arm, seeds 42,
43 and 44 are independent training seeds.

```bash
for seed in 42 43 44; do
  for arm in frozen unconditional selective conditional; do
    if uv run --no-sync python scripts/conditional_generator.py train \
      --data "$research_root/data" \
      --checkpoint checkpoint/generator \
      --arm "$arm" --seed "$seed" --device cuda \
      --batch-size 128 \
      --out "$research_root/warm/$arm/seed$seed"; then
      :
    else
      conditional_exit_code=$?
      if [ "$arm" = selective ] && [ "$conditional_exit_code" -eq 2 ]; then
        echo "Selective seed $seed has insufficient joint labels; inspect its metrics.json."
      else
        echo "Training failed: $arm seed $seed, exit $conditional_exit_code"
        break 2
      fi
    fi
  done
done
```

Defaults: one epoch training only the new modules, then at most ten full fine-tuning
epochs; new-module LR `1e-4`, backbone LR `1e-5`; sequence sampling first, followed
by an observed publication group; 75% annotated examples and 25% unconditional
replay; 20% observed-condition-token dropout; clipping at 1.0; early stopping after
three nonimproving joint epochs. The condition-only warmup applies to the conditional
arm. Other arms retain their existing architecture. Checkpoint selection uses
validation language-model NLL, never test-set predictor scores.

A run writes `run.json`, `history.csv`, `history.json`, `metrics.json`,
`checkpoint/{config.json,model.pt}` and `complete.json`. Check `status` before
sampling. The manifest records settings, data/model hashes, source provenance,
training curves, actual exposure and performance measurements.

Sample 10,000 raw draws per arm and generation seed:

```bash
for seed in 42 43 44; do
  for arm in frozen unconditional selective conditional; do
    model_dir="$research_root/warm/$arm/seed$seed/checkpoint"
    if [ ! -f "$model_dir/model.pt" ]; then
      echo "No checkpoint for $arm seed $seed; inspect its training metrics."
      continue
    fi
    if [ "$arm" = conditional ]; then
      uv run --no-sync python scripts/conditional_generator.py sample \
        --checkpoint "$model_dir" --label warm-conditional-request \
        --conditions experiments/conditional_selective_request_v1.json \
        --seed "$seed" --draws 10000 --batch-size 64 --device cuda \
        --out "$research_root/warm-draws/conditional-request-seed$seed" || break 2
      uv run --no-sync python scripts/conditional_generator.py sample \
        --checkpoint "$model_dir" --label warm-conditional-null \
        --seed "$seed" --draws 10000 --batch-size 64 --device cuda \
        --out "$research_root/warm-draws/conditional-null-seed$seed" || break 2
    else
      uv run --no-sync python scripts/conditional_generator.py sample \
        --checkpoint "$model_dir" --label "warm-$arm" \
        --seed "$seed" --draws 10000 --batch-size 64 --device cuda \
        --out "$research_root/warm-draws/$arm-seed$seed" || break 2
    fi
  done
done

uv run --no-sync python scripts/conditional_generator.py compare \
  --runs "$research_root"/warm-draws/* \
  --baseline warm-frozen \
  --activity-floor 0.8 --risk-ceiling 0.5 --device cuda \
  --out "$research_root/warm-comparison"

cat "$research_root/warm-comparison/REPORT.md"

uv run --no-sync python scripts/conditional_generator.py compare \
  --runs "$research_root"/warm-draws/conditional-* \
  --baseline warm-conditional-null --device cuda \
  --out "$research_root/condition-versus-null"
```

The frozen runs have identical weights; their different sample seeds measure
sampling variation, not three independently trained baselines. A trained arm uses
its matching training-seed checkpoint and the same-numbered generation seed. The
conditional NULL control uses the **same conditional checkpoint**, with no requested
assays, to test what the request itself changes.

The sampler uses full-prefix causal decoding, temperature 1.0, no top-k/top-p or
repetition penalty, and a fixed canonical alphabet/length mask. It retains every
draw in `raw.fasta` and `draws.csv`, including duplicates/reference matches.
`library.fasta` is the derived unique, exact-reference-free subset. All compared
runs must have identical sampling settings, budget and reference.

The comparison loads the frozen activity, hemolysis and panel heads; it requires
explicit per-head inference configs and records backbone revisions. For repeated
studies, pin the resolved 35M ESM revision using `--revision`. Inspect `scores.csv`
and `metrics.json` in each comparison cell, plus `comparison.json` and
`comparison.csv` at the comparison root.
The report and `comparison.json` also separate data/continued-training, selective
training, condition-request, AttnRes and MoE contrasts when matching metadata is
available. A request-versus-NULL contrast requires identical checkpoint weights
and configuration; numerical differences alone do not prove a biological effect.

## 5. Train matched architecture presets from scratch

These experiments isolate residual and feed-forward design from historical
checkpoint exposure. All presets use the same data partitions, condition schema,
sequence draws and number of training examples per epoch.

| Preset | Residual implementation | Feed-forward |
| --- | --- | --- |
| A | Standard | Dense, inner width 1536 |
| B | Corrected Block AttnRes v2, two layers/block | Dense, inner width 1536 |
| C | Standard | Four experts, top two, inner width 768/expert |
| D | Corrected Block AttnRes v2, two layers/block | Four experts, top two, inner width 768/expert |

All use six layers, width 384, six heads and biological cross-attention. Old
checkpoints retain legacy residual implementation v1. Full AttnRes v2 is available
as a tested correctness reference; it is not an additional default comparison arm.
MoE routes every valid token without dropping, uses FP32 probabilities, excludes
padding from differentiable balancing, and counts both chosen experts.

```bash
for seed in 42 43 44; do
  for preset in A B C D; do
    uv run --no-sync python scripts/conditional_generator.py train \
      --data "$research_root/data" \
      --arm conditional --preset "$preset" --seed "$seed" --device cuda \
      --batch-size 128 --epochs 100 \
      --out "$research_root/fresh/$preset/seed$seed" || break 2
  done
done
```

Fresh defaults: AdamW LR `3e-4`, weight decay `0.01`, 5% warmup then cosine decay,
clipping 1.0 and 100 epochs. Each preset completes the matched exposure budget;
validation chooses the saved checkpoint, while per-arm early stopping does not
shorten fresh runs. Compare recorded actual tokens, wall time, tokens/second,
peak GPU memory, total parameters, active feed-forward parameter estimates and
MoE auxiliary loss separately from language-model NLL. Matched active FFN work is
not a claim of equal total parameter count or equal measured runtime.

Generate requested and NULL outputs for each fresh model:

```bash
for seed in 42 43 44; do
  for preset in A B C D; do
    model_dir="$research_root/fresh/$preset/seed$seed/checkpoint"
    uv run --no-sync python scripts/conditional_generator.py sample \
      --checkpoint "$model_dir" --label "fresh-$preset-request" \
      --conditions experiments/conditional_selective_request_v1.json \
      --seed "$seed" --draws 10000 --batch-size 64 --device cuda \
      --out "$research_root/fresh-draws/$preset-request-seed$seed" || break 2
    uv run --no-sync python scripts/conditional_generator.py sample \
      --checkpoint "$model_dir" --label "fresh-$preset-null" \
      --seed "$seed" --draws 10000 --batch-size 64 --device cuda \
      --out "$research_root/fresh-draws/$preset-null-seed$seed" || break 2
  done
done

uv run --no-sync python scripts/conditional_generator.py compare \
  --runs "$research_root"/fresh-draws/* \
  --baseline fresh-A-request --device cuda \
  --out "$research_root/fresh-architecture-comparison"

uv run --no-sync python scripts/conditional_generator.py compare \
  --runs "$research_root"/warm-draws/frozen-seed* "$research_root"/fresh-draws/* \
  --baseline warm-frozen --device cuda \
  --out "$research_root/fresh-versus-incumbent"
```

The first comparison isolates architecture effects relative to standard dense
conditional training. The second checks whether those models improve over the
frozen primary incumbent. Warm-start results remain adaptation evidence because
historical pretraining may include the new held-out families.

## 6. Advance only passing three-seed candidates to the unchanged selector

The raw gate in `comparison.json` requires all seeds 42/43/44 with equal draw
budgets of at least 10,000. Joint unique yield must improve in each paired seed,
activity and every panel mean may fall by at most 0.03, mean pairwise distance by
at most 0.02, and valid unique novel fraction by at most 0.05. Risk strata compare
similar lengths and activity ranges. These are exploratory legacy-score gates,
not measured dose-specific hemolysis criteria.

Inspect the decisions:

```bash
uv run --no-sync python - "$research_root/warm-comparison/comparison.json" <<'PY'
import json, sys
for decision in json.load(open(sys.argv[1]))["decisions"]:
    print(decision["label"], "advance:", decision["advance_to_50000"])
    print(decision["seeds"])
PY
```

If `warm-conditional-request` passes, the following optional stage compares it
with the frozen primary at a matched **75,000 raw draws** per seed. At least
50,000 valid unique non-reference sequences are needed in **each** run. The selector
uses the first 50,000 clean candidates in draw order. If 75,000 draws are insufficient,
repeat every compared run at one larger common raw budget in new directories.
Do not compare 50,000 raw draws with a 50,000-clean-candidate library.

First make the gate concrete:

```bash
uv run --no-sync python - "$research_root/warm-comparison/comparison.json" <<'PY'
import json, sys
decisions = json.load(open(sys.argv[1]))["decisions"]
match = [d for d in decisions if d["label"] == "warm-conditional-request"]
assert len(match) == 1 and match[0]["advance_to_50000"], "Do not start the 50k stage: three-seed gate did not pass."
PY
```

Run the optional stage only after that assertion succeeds:

```bash
for seed in 42 43 44; do
  uv run --no-sync python scripts/conditional_generator.py sample \
    --checkpoint "$research_root/warm/frozen/seed$seed/checkpoint" \
    --label warm-frozen --seed "$seed" --draws 75000 --batch-size 64 --device cuda \
    --out "$research_root/large-draws/frozen-seed$seed" || break
  uv run --no-sync python scripts/conditional_generator.py sample \
    --checkpoint "$research_root/warm/conditional/seed$seed/checkpoint" \
    --conditions experiments/conditional_selective_request_v1.json \
    --label warm-conditional-request --seed "$seed" --draws 75000 --batch-size 64 --device cuda \
    --out "$research_root/large-draws/conditional-seed$seed" || break
done

uv run --no-sync python - "$research_root/large-draws" <<'PY'
import json, sys
from pathlib import Path
runs = sorted(Path(sys.argv[1]).glob("*"))
assert len(runs) == 6, "Expected three seeds for both models."
for run in runs:
    summary = json.loads((run / "summary.json").read_text())
    print(run.name, summary["unique_clean_candidates"])
    assert summary["unique_clean_candidates"] >= 50000, "Increase the common raw budget for all runs in new outputs."
PY

uv run --no-sync python scripts/conditional_generator.py compare \
  --runs "$research_root"/large-draws/* \
  --baseline warm-frozen --select-top --device cuda \
  --out "$research_root/large-top100-comparison"

cat "$research_root/large-top100-comparison/REPORT.md"
```

For a passing fresh preset, use the corresponding decision report, label and
checkpoint paths instead. Keep the same frozen baseline, three seeds, request and
common raw budget. The unchanged selector includes the current activity, conformity,
precision, breadth and MDR scores; hemolysis remains audit-only. Each cell writes
`library.fasta`, `top.fasta`, `top_scores.csv` and metrics including `library_complete`.
Require all libraries to be complete before interpreting this as a 50k comparison.

Report whether top-100 risk falls while activity/panel means stay within 0.03 and
pairwise distance within 0.02, across all three seeds. Any incomplete top-100 or
library is an unsuccessful comparison, not a releasable result.

## What to send back and what a result means

Send `support.json`, training `metrics.json`, the comparison `REPORT.md`,
`comparison.csv` and `comparison.json`. For a solver-free view of the generator,
inspect fresh-draw joint yield first; top-100 changes answer a separate selection
question. Do not cherry-pick one model seed or condition request after seeing test
scores.

Raw-draw confidence intervals describe fixed-model predictor-pass fractions.
Paired-seed bootstrap intervals describe these few training/sampling runs, not
biological success probabilities. The legacy hemolysis head is not a calibrated
concentration-specific assay predictor, so it cannot validate the requested 10%
lysis threshold at 64 µM. Lower risk scores require independent follow-up evidence.

No operation here updates `checkpoint/generator`, `checkpoint/generator_blend`,
reward heads, default blending, release FASTAs, or official submission records.
Architecture complexity and lower language-model loss are not sufficient for
promotion. A candidate still needs cross-seed comparison and separate release review.
