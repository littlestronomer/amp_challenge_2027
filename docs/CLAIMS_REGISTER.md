# Submission claims register

This register tracks the claims in `ABSTRACT.md`. A file hash or validation
pass establishes artifact identity or technical execution; it does not establish
biological correctness.

| Claim | Source | Population / instrument | Limitation | Disposition |
|---|---|---|---|---|
| Two-generator 3:1 interleave is the intended library recipe | `PHASE1_RESULTS.md`; `src/amp_challenge_2027/generate.py` | Seed 42, shipped checkpoints | Must be rechecked by current strict run and manifest | Retain as intended recipe; strict check pending |
| Generator has 6 × 384 configuration and 10.7M parameters | `checkpoint/generator/config.json`; model config | Shipped primary model | Count/config should be recalculated from artifact if quoted in external material | Retain with source file |
| Local FBD 0.221 and MMD 0.357 | `PHASE1_RESULTS.md` | ESM-2 650M and recorded seqme protocol | Historical result; not the hidden aggregation score; source artifacts not yet re-inventoried | Retain only as historical protocol-specific values |
| Recall 0.899 and Precision 0.863 | `PHASE1_RESULTS.md` | ESM-2 650M library evaluation | Does not measure wet-lab activity or out-of-family generalization | Retain only with explicit caveat |
| Top-100 predicted broad activity | Historical top-100 audits and `checkpoint/reward/classifier_panel.pt` | Genus-level classifier | Model outputs; validation overlap and selected-sequence generalization limit interpretation | Reword as model-derived ranking signal |
| Deployed panel AUROC 0.81 / binary 0.786 / hemolysis 0.846 | Earlier abstract and audit records | Frozen heads with random split/reconstructed validation | Does not establish independent family generalization; newer benchmark status needs remote artifact verification | Do not quote in abstract pending evidence inventory |
| Homolog leakage bounded at −0.055 | Earlier abstract | Fine-tuned reference model vs another split | Not a bound for the deployed heads or the selected candidates | Remove |
| No concentrated hemolysis risk | Earlier abstract | Top-100 hemolysis head | Surrogate result, unclear current evidence and not experimental safety | Remove |
| Library is reproducible and submission-valid | `PHASE1_RESULTS.md` | Cold clone; two runs; 2026-09-05 | Historical; current strict run pending | Label as historical until reverified |
| Data provenance and licenses are complete | `docs/DATA_DISCLOSURE.md` | Declared sources and files | Verify actual distributed snapshots/terms before public release | Pending release audit |

Update this table whenever a numerical or comparative claim changes. Include the
artifact path/hash, evaluation population, model identity, protocol and known
overlap in any claim proposed for the abstract. Do not convert missing fields
into favorable claims.
