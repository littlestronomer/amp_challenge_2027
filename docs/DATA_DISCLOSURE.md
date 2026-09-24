# Training-Data Disclosure

Declared accounting of data sources used to build the submitted
artifacts, per the competition's full-track disclosure requirement. Sources
split into **final-stack** (data that trained the shipped models) and
**development-only** (used in experiments/selections recorded in
PHASE1_RESULTS.md but not in the shipped training paths). The SSH
`data/raw/sources.json` supplied by the author on 2026-09-24 records only
four DRAMP downloads. It is not a complete provenance registry for the
deployed models. Reported DRAMP SHA-256 values are quoted below.

This disclosure records declared provenance; it does not mean that every
training snapshot is included in this checkout or may be redistributed. Verify
each artifact's actual input hashes and applicable source terms before treating
the repository as a complete public data release.

## Final-stack training data

Author declaration (2026-09-24): no private training data were used. The author
has not confirmed that the public sources listed below are exhaustive. Recover
the actual SSH snapshots and source records before signing off the full disclosure.

| artifact trained | data | source & license | retrieved |
|---|---|---|---|
| Generators (`checkpoint/generator`, `checkpoint/generator_blend`) | 39,448 curated peptides (`data/processed/generative.csv`), standard AAs, length 8–50, deduplicated | **MarLys** (Mendeley `w4hb5grjwb`), CC-0, per the AMP Challenge starter data | provided with the competition repository |
| Panel ranker (`checkpoint/reward/classifier_panel.pt`) | 69,461 strain-level MIC rows → 39,216 per-genus rows → 13,485 labeled sequences (`data/processed/activity_labels_full.csv`) | **DBAASP v4** via public REST API (`dbaasp.org/peptides`, 25,069 peptide cards), API data-usage policy; cite Pirtskhalava et al., *NAR* 49(D1):D288–D297 (2021) | 2026-08-28 |
| Binary activity head (`checkpoint/reward/classifier.pt`) | 1,423 binarized labels (legacy set) | same DBAASP lineage | 2026-08 (legacy build) |
| Hemolysis head (`checkpoint/reward_hemo/classifier.pt`) | 4,752 labels from erythrocyte (concentration, lysis-band) rows | **DBAASP v4** (as above) | 2026-08-29 |

The reference/evaluation set `data/antibacterial.fasta` (39,448 sequences)
ships with the competition tooling and is used only for filtering, novelty
screening, and protocol evaluation — never as held-out labels. Note it
coincides with the MarLys training corpus (discussed as the in-sample
framing in PHASE1_RESULTS.md); submission-side novelty/identity filters
enforce the emission rules.

## Development-only data (not in shipped training paths)

| data | used for | source & license |
|---|---|---|
| DRAMP 3.0 splits (general/antibacterial/anti-Gram±, FASTA) | expanded-corpus experiment (null result, recorded); Gram-membership label exploration | **DRAMP 3.0**, CC BY 4.0; Kang et al., *Sci Data* 6:170 (2019); retrieved 2026-08-27 via `download.php` |
| DRAMP `general_amps.xlsx` (activity text) | MIC-clause extraction experiments (converter kept, output unused by final models) | as above |
| DBAASP hemolysis rows (23k) | safety audit of the top-100 (head deployed at weight 0) | DBAASP v4 (as above) |

DRAMP SHA-256 (transcribed from the author's SSH `data/raw/sources.json`,
2026-09-24; actual raw-file bytes still require verification on SSH):
`antibacterial_amps.fasta` a0d484eb6176e123298d19d06f23ccc6398a1f5c75d169cccbbf7590a6e6c835;
`general_amps.fasta` 5915e91b3501c41a914a05403dfd2a435af59116ffb738a8423d34995a2b9c26;
`anti_gram_negative.fasta` 3c1b5d0fb323baf6d6c0d8d8164d8b35b3bbc41c9296b0665894f99fb4f12c41;
`anti_gram_positive.fasta` 348cada15ae82f593af683ee395b35bc61df81dddfcc7041f8118838bacee8e0.

The general and anti-Gram-negative hashes above correct differences between the
previous disclosure and the supplied manifest; this is a transcription correction,
not an independent validation. `scripts/audit_authorship_readiness.py` checks
manifest hashes and sizes against local raw files. The download registry is
written by the DRAMP fetch path; its lack of MarLys/DBAASP entries does not prove
those datasets were unused. Processed-file hashes alone also cannot establish
which snapshot trained each checkpoint.

## Pretrained components

| component | model | license |
|---|---|---|
| Classifier backbones (activity/panel/hemolysis) | `facebook/esm2_t12_35M_UR50D` | MIT (HF) |
| Precision proxy + local eval embedders | `facebook/esm2_t6_8M_UR50D`, `facebook/esm2_t33_650M_UR50D` | MIT (HF) |

## Reproducibility statement

Fetch implementations exist (`scripts/fetch_data.py`, `scripts/fetch_dbaasp_v4.py`),
but complete snapshot-to-checkpoint provenance has not been recovered. Label construction is scripted and tested
(`scripts/build_ranking_labels.py`, `scripts/build_hemolysis_labels.py`);
canonical corpora are hash-frozen by `scripts/rebuild_corpus.py`. `uv run
generate` (defaults, seed 42) reproduces the submitted library and top-100
byte-for-byte in historical reported checks (see PHASE1_RESULTS.md).
These historical checks do not certify the final release commit. Current
release gaps and snapshot reconciliation are tracked in
`docs/AUTHORSHIP_READINESS.md`; no claim of complete authorship eligibility is made.
