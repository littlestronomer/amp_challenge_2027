# Competition status template

This is a placeholder for the SSH evidence review. Run
`docs/RUNBOOK_COMPETITION_READINESS.md` and replace each `PENDING` item with a
source-linked finding. Until then, the scientific decision is **pending**.

## What is established

- The repository contains trained generator and scorer artifacts and historical
  submission validation/results in `PHASE1_RESULTS.md`.
- The current default library recipe is the 3:1 incumbent blend. The ranking
  combines model-derived and property-derived signals; hemolysis weight is zero.
- Historical abstract statements about safety, calibration and homolog leakage
  have been removed pending artifact-level review.

## SSH evidence review

| Area | Result | Source artifact | Interpretation |
|---|---|---|---|
| Strict generation/manifest | PENDING | | |
| Inference metadata parity | PENDING | | |
| Family-held-out activity/panel/hemolysis | PENDING | | |
| Label/metadata reconstruction | PENDING | | |
| Label ablation | PENDING | | |
| Incumbent top-100 audit | PENDING | | |
| Incumbent vs candidate selector comparison | PENDING | | |
| External complete-pipeline comparison | PENDING | | |
| Official score and rank | PENDING / unavailable | | |

## Decision

**PENDING.** First verify the artifacts above. Do not choose a generator, selector,
or predictor intervention from this template. No numerical likelihood of
advancing to wet-lab testing is assigned.

## Next action

Run the commands in `RUNBOOK_COMPETITION_READINESS.md`, copy the generated
`STATUS.md` and small summary artifacts back for review, and update this table
with exact source hashes and limitations.
