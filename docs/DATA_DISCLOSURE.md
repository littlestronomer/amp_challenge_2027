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

The [lineage ledger](release/TRAINING_LINEAGE.json) pins the deployed models and
candidate CSV hashes; [training provenance](TRAINING_PROVENANCE.md) distinguishes
those identities from missing historical training links. Current primary-source
terms and acknowledgements are in [third-party notices](THIRD_PARTY_NOTICES.md).

The author-supplied SSH lineage report confirms all 13 MarLys upstream database
names occur in the current generator CSV's source annotations, including DRAMP
and DBAASP. These overlapping annotations are recorded in
[SSH_LINEAGE_REPORT.json](release/SSH_LINEAGE_REPORT.json). “Development-only
DRAMP” below means the separate downloads, not absence of indirect DRAMP ancestry.

## Final-stack training data

Author declaration (2026-09-24): no private training data were used. The author
has not confirmed that the public sources listed below are exhaustive. Recover
the actual SSH snapshots and source records before signing off the full disclosure.

| artifact trained | data | source & license | retrieved |
|---|---|---|---|
| Generators (`checkpoint/generator`, `checkpoint/generator_blend`) | 39,448 curated peptides (`data/processed/generative.csv`), standard AAs, length 8–50, deduplicated | **MarLys** (Mendeley `w4hb5grjwb`), CC-0, per the AMP Challenge starter data | provided with the competition repository |
| Panel ranker (`checkpoint/reward/classifier_panel.pt`) | Current candidate CSV: 39,216 rows, 11,402 unique sequences before loader filtering/aggregation (`data/processed/activity_labels_full.csv`) | **DBAASP v4** via public REST API; historical notes report 25,069 peptide cards and 69,461 strain-level MIC rows, not independently recounted here. Cite Pirtskhalava et al., *NAR* 49(D1):D288–D297 (2021) | 2026-08-28 (historical declaration) |
| Binary activity head (`checkpoint/reward/classifier.pt`) | Current candidate CSV: 2,009 rows, 1,423 unique sequences (`data/processed/activity_labels.csv`), before loader filtering | same declared DBAASP lineage | 2026-08 (legacy build) |
| Hemolysis head (`checkpoint/reward_hemo/classifier.pt`) | 4,752 labels from erythrocyte (concentration, lysis-band) rows | **DBAASP v4** (as above) | 2026-08-29 |

The 2026-09-25 SSH report supersedes the earlier panel count of 13,485 for this
hash-pinned CSV. Unique sequences, CSV rows and records admitted by the training
loader are different quantities. The binary loader retains recognized-label rows;
the panel loader filters labels/organisms/sequences and aggregates by sequence.
These CSV summaries do not establish original training population or split sizes.

The reference/evaluation set `data/antibacterial.fasta` (39,448 sequences)
ships with the competition tooling and is used only for filtering, novelty
screening, and protocol evaluation — never as held-out labels. Note it
coincides with the MarLys training corpus (discussed as the in-sample
framing in PHASE1_RESULTS.md); submission-side novelty/identity filters
enforce the emission rules.

DBAASP hemolysis measurements trained the separate classifier listed above:
4,752 derived labels were used for this head. Earlier notes mention approximately
23k raw rows; this is not an additional training corpus or the final label count.
The head supports computational audits, and its predictions have zero weight in
final selection. "Audit-only" describes use of the trained head, not absence of
training on hemolysis labels. These labels were not a training objective for
either generator.

## Development-only data (not in shipped training paths)

| data | used for | source & license |
|---|---|---|
| Separate DRAMP splits (general/antibacterial/anti-Gram±, FASTA) | expanded-corpus experiment (null result, recorded); Gram-membership label exploration | CC BY 4.0; URLs use `DRAMP3.0_new`; cite Shi et al., DOI 10.1093/nar/gkab651 for DRAMP 3.0. Retrieved 2026-08-27; URL names do not certify a frozen database version. |
| DRAMP `general_amps.xlsx` (activity text) | MIC-clause extraction experiments (converter kept, output unused by final models) | as above |

DRAMP SHA-256 (transcribed from the author's SSH `data/raw/sources.json`,
2026-09-24; the author supplied a successful raw-file hash/size audit on 2026-09-25):
`antibacterial_amps.fasta` a0d484eb6176e123298d19d06f23ccc6398a1f5c75d169cccbbf7590a6e6c835;
`general_amps.fasta` 5915e91b3501c41a914a05403dfd2a435af59116ffb738a8423d34995a2b9c26;
`anti_gram_negative.fasta` 3c1b5d0fb323baf6d6c0d8d8164d8b35b3bbc41c9296b0665894f99fb4f12c41;
`anti_gram_positive.fasta` 348cada15ae82f593af683ee395b35bc61df81dddfcc7041f8118838bacee8e0.

The general and anti-Gram-negative hashes above correct differences between the
previous disclosure and the supplied manifest; this is a transcription correction,
followed by the reported SSH audit, not an independent local validation. `scripts/audit_authorship_readiness.py` checks
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
On 2026-09-25, the author supplied a fresh-clone validator pass at commit
`be3aecae7cdde9bc898337c3b282d3fa4714457f` and matching library/top hashes against
the frozen candidate. This is recorded in `docs/release/BASELINE.json` with
evidence attribution. A further author-run SSH validation on 2026-09-30 at
`7a5b86bf11f43a207a2a2c7f2b9c0fd82c22169d` passed two default runs and a strict run,
with both FASTAs matching those historical hashes. The downloaded evidence and
FASTA hashes were checked locally; see [the validation record](release/VALIDATION_2026-09-30.json).
These inference checks do not resolve original training-input or split lineage.
Release status and remaining disclosure gaps are tracked in
[RELEASE_CANDIDATE.md](RELEASE_CANDIDATE.md). No claim of complete authorship
eligibility is made.
