# Competition status

## 2026-09-25 release evidence

Author-supplied SSH output reports default fresh-clone validation and two-run
byte equality at `be3aecae7cdde9bc898337c3b282d3fa4714457f`. Library and top hashes
also match the frozen competition-readiness candidate. All inventoried release
files match Git; four DRAMP downloads match their recorded hashes/sizes.
See `release/BASELINE.json` and `RELEASE_CANDIDATE.md`. These results close the
metadata and reported reproducibility gaps for that baseline, not historical
training lineage or biological validation. GitHub remains private as checked
on 2026-09-25. Entries below retain their historical evidence dates.

## 2026-09-24 correction: constrained selector and release eligibility

The initial constrained selector accidentally reused a dense risk row for its
first pairwise constraint. All prior constrained-selection infeasibility claims
are withdrawn, including the strict risk ceilings. Incumbent fallback demonstrated
baseline feasibility only; it did not show reranking is exhausted. Use the
repaired selector runbook and fresh output directories. Generator/ranker
promotion is not supported by those earlier results.

GitHub visibility was checked: repository private; default branch `main`.
Co-authorship eligibility is not yet established. See `AUTHORSHIP_READINESS.md`
for missing predictor metadata, data provenance and final-clone validation steps.
The historical assessment below remains a record of its stated evidence date.

Updated 2026-09-21 from repository inspection and SSH observations pasted by the
user. Remote measurements below are reported evidence, not artifacts independently
verified in this checkout. Use `experiments/competition_evidence_v2.json` to
inventory the files after synchronizing them.

## Current assessment

| Area | Current position | Evidence and limit |
|---|---|---|
| Engineering | Promising; strict generation reportedly ran twice with matching library, top and score hashes. | The pasted hashes establish same-machine byte integrity as reported. Persist both runs and a comparison marker for an auditable two-run check. |
| Generator | Keep the incumbent 3:1 blend. | Current top hash reportedly matches historical hybrid seed 42 exactly; no evidence supports generator retraining or changing checkpoints. |
| Selector | Keep the incumbent ranking. | Historical normalization comparison reportedly retained all 100 incumbent members and did not show a compelling improvement from P1. |
| Deployed predictor evidence | Weak to moderate retrospective evidence; insufficient for promotion. | Production inference probes reportedly match reconstruction outputs. Reconstructed hemolysis metrics have substantial near-neighbor overlap and unresolved chemistry/provenance concerns. |
| Family benchmark | No clear MLP advantage over the linear baseline. | All three paired architecture-difference intervals reportedly include zero; these tests are already inspected. |
| Generated candidates | Plausible model-ranked candidates; no experimental result. | Reported Top-100 audit: 100 valid unique members, no exact reference overlap, maximum similarity 0.8, no pairwise ratio at or above 0.8. Scores are surrogates; synthesis and activity are unassessed. |
| Wet-lab selection | Unknown. | No independently validated candidate ranking, synthesis assessment, or wet-lab measurements are available. No selection probability is assigned. |
| Competition standing | Unknown/unavailable. | No official submission receipt, score, or rank is configured. |

The hemolysis head's positive output is risk under its candidate label definition,
not percent lysis. Panel breadth/MDR values are fractions of probabilities over
0.5, not confidence. A reconstructed AUROC or parity probe cannot establish
generalization to the selected peptides.

## Decision

Retain the incumbent generator and ranking. Do not change model weights or
promote a risk-aware selector from surrogate scores. The predeclared frozen-pool
comparison in `experiments/selectivity_tradeoff_v1.json` is authorized as a
sensitivity description: it can show whether this selector retrieves lower
predicted-risk alternatives in the same pool and the associated surrogate-score
trade-offs. That is compatible with weak predictor generalization because the
experiment evaluates selector behavior and candidate availability, not predictor
truth. Even a pass only supports considering a separate independent evaluation.

## Remaining evidence

1. Persist and inventory both strict-run output directories and the byte-comparison
   report if a durable repeatability record is required.
2. Run the bounded GPU-1 cache preflight and risk inference in
   `docs/RUNBOOK_SELECTIVITY_TRADEOFF.md`; review the complete CPU comparison.
3. Resolve label units/thresholds, nonstandard chemistry and original split/data
   provenance before further predictor training.
4. Add an official submission receipt if one exists; otherwise leave rank unknown.
