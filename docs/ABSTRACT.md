# Submission Abstract — AMP Challenge 2027

## Method summary

The submission uses two autoregressive peptide generators and a deterministic
3:1 interleave of their candidate pools. The primary decoder has six layers,
384 hidden units, block-wise attention residuals and 10.7 million parameters.
Both generators were trained on the competition-provided antibacterial corpus.
The secondary generator conditions on peptide charge bins sampled from the
reference distribution. The intended 50,000-sequence library is produced with
seed 42.

The top-100 selector combines five model-derived or property-derived signals:
binary activity, genus-level panel breadth, MDR-genus breadth, property
conformity and an embedding-space precision proxy. Selection also applies
sequence-validity, exact-overlap, novelty and diversity rules. These scores are
ranking surrogates. They are not measurements of antimicrobial potency,
strain-level activity, hemolysis or clinical safety.

Historical local evaluation recorded ESM-2 650M embedding-space FBD 0.221 and
MMD 0.357 for the blended library, with Recall 0.899 and Precision 0.863 under
the recorded seqme protocol. These characterize similarity to the reference AMP
distribution under that protocol; they are not the competition's hidden
aggregation score or evidence of experimental activity. The training corpus
coincides with the competition reference corpus. Submission filters enforce the
competition's sequence overlap and top-candidate similarity limits.

Historical records report a successful cold-clone, two-run reproducibility and
submission-format validation on 2026-09-05. A current strict release validation
and source-artifact audit are pending. The deployed activity heads were trained
from DBAASP-derived measurements; their validation splits, family overlap,
calibration and label provenance are described in the audit records. Those
results do not establish performance on unseen peptide families or on the
selected generated peptides. Hemolysis is not used in the default ranking
recipe, and no safety conclusion is made here.

Complete source and license details are in `docs/DATA_DISCLOSURE.md`. Historical
experiments, including negative and superseded results, are retained in
`PHASE1_RESULTS.md`. Numerical results should be quoted together with the
corresponding model artifact, population, split and protocol from their source
report.
