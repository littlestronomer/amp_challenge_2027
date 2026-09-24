# littlestronomer: release and authorship readiness

Audit date: 2026-09-25. Author: Göktürk Batın Dervişoğlu;
correspondence: dervisoglu21@itu.edu.tr. Affiliation has not been supplied.
The author confirmed on 2026-09-24 that no private data were used, but could
not confirm that the named public datasets are the complete source inventory.

Goal: a compliant submission eligible for the planned joint benchmarking paper.
This is not a guarantee of wet-lab admission, individual author-list inclusion,
or publication acceptance. Do not mistake Kaggle Kudos for an authorship decision.

## Official basis

The [official template](https://github.com/szczurek-lab/amp-challenge-2027)
requires a public repository, weights, inference code, usage documentation,
permissive license, default reproducibility, and complete training-data disclosure
for co-authorship eligibility. The [competition overview](https://www.kaggle.com/competitions/amp-challenge/overview)
distinguishes these full requirements from minimum benchmark participation.
See the organizers' current rules when submitting; no publication guarantee is inferred.

## Findings

| Requirement | Evidence found | Remaining work |
|---|---|---|
| Public repository | Authenticated GitHub lookup on 2026-09-25 returned `isPrivate: true` | Review release contents, then explicitly approve public visibility before final submission. No visibility change was made. |
| Release branch | GitHub default is `main`; current work is `feat/nway-blend-multiaxis-conditioning` | Validate and designate the branch/commit delivered to organizers; default-clone behavior matters. |
| Author and method documentation | Identity file, abstract, data disclosure, README | Confirm name spelling/affiliation and any additional contributors before submission. |
| Permissive code license | BSD-3-Clause present, retaining upstream copyright | Preserve third-party notices; source data rights are separate from code licensing. |
| Weights | Both generators, activity/panel/hemolysis weights are Git-tracked | SSH audit confirms every listed weight/config matches HEAD; hashes are preserved in release/BASELINE.json. |
| Predictor metadata | Three per-head configs transcribed from the author's SSH output; inference fields match the registry for the actual committed checkpoint hashes | SSH backup and semantic comparison succeeded; all three files match the originals. Historical `val_auroc` values are retained as supplied, not independently revalidated. |
| Default reproducibility | Author reports fresh-clone default validator pass at be3aeca, two-run equality and exact frozen FASTA matches | Preserve this baseline; compare runtime hashes for subsequent release changes. |
| Training data disclosure | MarLys aggregate and DBAASP declared for deployed stack; separate DRAMP downloads declared development-only; author confirms no private data | Reconcile original training files, splits and checkpoint hashes. Source records are absent locally and ignored by Git. The full public-source inventory is not yet confirmed. |
| Submission/admission | No receipt or organizer decision supplied | Submit once the release is checked and retain the receipt. Do not submit repeated entries for the same model contrary to competition rules. |

## Dataset trace

- Generator and charge-conditioned generator: declared competition MarLys corpus;
  `data/processed/generative.csv`. Both checkpoint configs match the documented
  six-layer, 384-unit decoder; the secondary config has charge conditioning.
- Binary activity: declared legacy DBAASP-derived 1,423 labels. Candidate `activity_labels.csv` is pinned by the provisional reconstruction protocol; original training-time hash and split remain unverified.
- Panel classifier: declared DBAASP-derived `activity_labels_full.csv`, 13,485
  labelled sequences after aggregation. Current snapshot hash matches the provisional reconstruction protocol; original training-time linkage remains needed.
- Hemolysis classifier: declared 4,752 DBAASP-derived labels; candidate `hemolysis_labels.csv` hash is pinned and reported on SSH; original training-time linkage remains unresolved.
- Separate DRAMP downloads: declared development experiments. MarLys may supply indirect DRAMP-derived sequences; actual subset source annotations still require inspection.
- ESM backbones: pretrained public components; include model identifiers/revisions
  and their licenses in the final disclosure.

The supplied SSH download manifest lists four DRAMP files only. The source
fetcher records DRAMP downloads there; the list is not an exhaustive training
inventory. The audit checks their actual hashes/sizes and inventories the
processed training-file hashes, without claiming they identify checkpoint inputs.
DRAMP registry keys such as `dramp-general/general_amps.fasta` are logical source
IDs; the downloader stores those files under `data/raw/dramp/`. The audit now
reports both the registry key and local path. Audits from commit `0994940` checked
the logical ID as a path. The author reran v3 and all four actual files verified.

`docs/DATA_DISCLOSURE.md` is a lineage declaration, not proof that each checkpoint
used exactly those files. Do not relicense downloaded public databases under the
repository BSD license. Verify their terms and document access/reconstruction;
any genuinely non-public training data triggers the organizers' separate public
release requirement.

## Run the audit on SSH

```bash
uv run --no-sync python scripts/audit_authorship_readiness.py \
  --out sweep_results/authorship-readiness-v4
cat sweep_results/authorship-readiness-v4/REPORT.md
```

If `gh` is installed/authenticated, optionally add
`--github-repo littlestronomer/amp_challenge_2027` to check visibility/default branch.
Use a new output directory after changes. The audit only inventories files; it
does not upload data, publish the repository, or certify biological performance.

The three predictor configs now match their preserved SSH originals and are
committed. Default fresh-clone validation and exact FASTA comparison passed as
reported by the author. Continue with the CPU collector in
[TRAINING_PROVENANCE.md](TRAINING_PROVENANCE.md); see the reviewed baseline and
publication sequence in [RELEASE_CANDIDATE.md](RELEASE_CANDIDATE.md).

## Final release verification (after resolving artifact gaps)

On GPU 1, validate the intended branch in a new clone:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/verify_submission.py \
  https://github.com/littlestronomer/amp_challenge_2027 \
  --branch feat/nway-blend-multiaxis-conditioning \
  --dir submission/release-default-v1
```

No `--extra` and no `--strict-generation` here: this checks the organizers'
default inference path. The validator performs synchronization in the fresh
clone and compares two runs. Compare the resulting `generate/library.fasta`
and `generate/top.fasta` hashes to the exact files selected for submission;
same-clone repeatability alone does not establish submission equality.
Also run the strict release checks in `RUNBOOK_COMPETITION_READINESS.md`.

Publication of this private repository requires a final contents review and
explicit approval. Registration/submission and contacting organizers are separate
external actions, not performed by this audit.
