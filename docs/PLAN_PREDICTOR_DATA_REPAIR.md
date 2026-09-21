# Predictor data repair plan

Status: **design only; no training or checkpoint promotion authorized.**

Reported predictor evidence is limited by near-neighbor overlap in reconstructed
validation, nonstandard chemistry in hemolysis records, unresolved label
threshold/units, and uncertain original split provenance. Repair and document the
observations before fitting a new candidate. Continue using existing
observation-audit, metadata-reconciliation, molar-label and family-benchmark
tools where their schemas fit.

## Data contract

Preserve each source observation as received and retain a stable source record
identifier, source URL/license where available, sequence string, chemistry or
modification description, assay name, organism/strain, units, tested
concentration, inequality/censoring operator, replicate identity, assay date,
and explicit label-building provenance. Derived labels must point back to all
observations and rules that produced them.

Partition records into supported canonical unmodified linear sequences, known
modified chemistry, unknown chemistry, and malformed records. Do not silently
strip residues, canonicalize distinct molecules into the same string, convert a
nonstandard symbol into another amino acid, or discard a record without an
auditable reason. Keep modified/unknown records available for separate review;
do not mix them into a model population whose tokenizer/labels cannot represent
their chemistry.

Resolve discordant and censored observations with predeclared assay-aware rules.
Missing results mean unknown, never negative. Do not ask a language model to
decide biological labels. Report excluded, unresolved and conflicting records
by source and reason.

## Split and evaluation design

Deduplicate and group sequence families before split assignment. Keep close
homologs and replicate observations within one split, then audit exact and
near-neighbor overlap across training, validation, calibration and test. Preserve
the family map and split assignment as hashed artifacts.

Reserve a calibration partition independently from model fitting and model
selection. The already inspected reconstruction and family-test results remain
historical descriptive evidence. A future confirmatory claim requires genuinely
new compatible observations or a declared nested design that accounts for all
selection.

Compare the current linear baseline and current-size MLP on the same examples,
labels, family split and tuning budget. Record dataset and split hashes, model
recipe/checkpoint hashes, uncertainty and data support near each candidate, AUROC,
average precision, Brier score, log loss, calibration error and subgroup results.
Do not assume that a larger model will improve performance.

## Promotion gate

Promote a predictor or selector only with matched-population evidence and an
appropriate independent assessment of the proposed candidates or selection
procedure. Optimization scores on the same predictions are descriptive only.
Document calibration limits, family support, label uncertainty and chemistry
coverage. A separate user-approved training protocol is required before any new
training sweep or checkpoint replacement.
