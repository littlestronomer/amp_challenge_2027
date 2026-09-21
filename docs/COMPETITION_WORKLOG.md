# Competition readiness worklog

Implementation started 2026-09-21 from baseline `ccf8ef6`.

## Baseline snapshot

- Existing local edits before this work: `pyproject.toml`, `uv.lock` (Jev SDK
  optional extra). Preserved.
- Incumbent generation: `checkpoint/generator` plus
  `checkpoint/generator_blend`, interleave ratio 3:1, seed 42.
- Default ranker weights: activity 0.5, conformity 0.25, precision 0.25,
  breadth 1, MDR breadth 1, hemolysis safety 0.
- Reference: `data/antibacterial.fasta`.
- Historical technical validation: recorded in `PHASE1_RESULTS.md` on
  2026-09-05. This is not a current strict run or leaderboard score.
- Latest SSH experiment artifacts are not present in this local checkout at
  implementation time. The collector records them as missing rather than
  reconstructing their contents.

## Implementation checklist

- [x] Add explicit, CPU-only evidence inventory and status output.
- [x] Add opt-in strict incumbent generation and provenance manifest.
- [x] Add sequence-keyed top-100 score output to strict runs.
- [x] Add whole-top-100 novelty/redundancy/prediction audit and 25-subset
      conditional score diagnostic.
- [x] Correct abstract claims and add a claim register.
- [x] Add SSH readiness runbook, remote decision template, and independent-data
      protocol gate.
- [x] Add synthetic checks for marker integrity, missing roots, traversal,
      unsupported inputs, deterministic inventory and top-100 score coverage.
- [ ] Run `--strict` generation on SSH and review its manifest.
- [ ] Run top-100 readiness audit on the generated artifacts.
- [ ] Collect the latest remote benchmark/audit outputs and review evidence.
- [ ] Choose and run one justified next experiment.
- [ ] Verify current official score/submission receipt, if a submission exists.

## Current blockers

1. SSH-only results are not in this checkout, so no scientific intervention is
   selected from unverified summaries.
2. No official Kaggle submission receipt/score is configured in the evidence
   inventory.
3. No independent external predictor dataset was selected and verified for
   task compatibility and reuse rights. Do not call a new split of the current
   inspected data independent.

## Local implementation verification — 2026-09-21

- `pytest -q tests/test_competition_readiness.py tests/test_pipeline.py`: 12
  passed.
- Ruff passed on all changed Python files; compileall and `git diff --check`
  passed.
- A local evidence inventory completed with no invalid or required-missing
  sources. Fourteen optional SSH/result roots are absent from this checkout,
  including family-held-out fits/test, strict generation, selection audits and
  any official submission receipt.
- Strict generation, GPU inference and competition experiments were not run
  locally. Use the SSH runbook after pulling this branch.

## Work entries

Add dated entries below with actual commands, code revision, source hashes,
results and limitations. Do not overwrite old experiment evidence.

## Selectivity reliability implementation — 2026-09-21

- Implementation commit: `a444a5d` on
  `feat/nway-blend-multiaxis-conditioning`.

- Fixed ordinary repeatability validation to retain first-run bytes before the
  second run overwrites the shared output directory. Added focused mocked-file
  regression tests; strict mode still compares separate output directories and
  score CSV bytes.
- Added evidence inventory v2 configuration. It points to the completed
  `selection-normalization-v1` producer layout, retains the legacy
  selection-diagnosis source separately, and records reconstruction,
  predictor-artifact, strict-generation and strict-repeatability evidence as
  distinct sources. Inventory roots now mean `present`; hashes/markers determine
  integrity statuses. Expected-hash disagreement is invalid.
- Generation provenance now uses manifest schema v2 and distinguishes neural
  backbone applicability, a runtime-resolved revision and an explicitly pinned
  load. It records effective config source/hash, precision cache identity,
  logical device and `CUDA_VISIBLE_DEVICES`. Old manifest v1 remains readable.
- Resume mismatch errors now report bounded differing recipe key paths and
  categories while preserving the refusal to mix experiments.
- Added a frozen C0/R1 protocol, resumable sequence-keyed hemolysis risk cache,
  CPU-only selection comparison and SSH runbook. The risk cache and comparison
  have not run on SSH; no new result or promotion is claimed.
- Added a predictor-data repair specification. No training, checkpoint change,
  generator change or default safety weight was introduced.
- Local tests were prepared but intentionally not run in this implementation
  session. SSH inference and scientific outcome review remain pending.
