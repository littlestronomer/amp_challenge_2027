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
