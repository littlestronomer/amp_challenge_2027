# littlestronomer: release and authorship readiness

Audit date: 2026-09-24. Author: Göktürk Batın Dervişoğlu;
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
| Public repository | Authenticated GitHub lookup on 2026-09-24 returned `isPrivate: true` | Review release contents, then explicitly approve public visibility before final submission. No visibility change was made. |
| Release branch | GitHub default is `main`; current work is `feat/nway-blend-multiaxis-conditioning` | Validate and designate the branch/commit delivered to organizers; default-clone behavior matters. |
| Author and method documentation | Identity file, abstract, data disclosure, README | Confirm name spelling/affiliation and any additional contributors before submission. |
| Permissive code license | BSD-3-Clause present, retaining upstream copyright | Preserve third-party notices; source data rights are separate from code licensing. |
| Weights | Both generators, activity/panel/hemolysis weights are Git-tracked | Compare SSH artifact hashes against the committed release. |
| Predictor metadata | Three per-head config files are absent in this local checkout and committed tree | Retrieve the verified SSH files and reconcile with frozen-head audit. Do not synthesize temperatures or change layer metadata to pass a check. |
| Default reproducibility | Entry point, Python version, lockfile and two-run validator exist; historical SSH repeatability was reported | Cold clone the exact final commit, run plain `uv sync` and default generation twice, compare to the submitted library/top bytes. |
| Training data disclosure | MarLys and DBAASP declared for deployed stack; DRAMP declared development-only; author confirms no private data | Reconcile original training files, splits and checkpoint hashes. Source records are absent locally and ignored by Git. The full public-source inventory is not yet confirmed. |
| Submission/admission | No receipt or organizer decision supplied | Submit once the release is checked and retain the receipt. Do not submit repeated entries for the same model contrary to competition rules. |

## Dataset trace

- Generator and charge-conditioned generator: declared competition MarLys corpus;
  `data/processed/generative.csv`. Both checkpoint configs match the documented
  six-layer, 384-unit decoder; the secondary config has charge conditioning.
- Binary activity: declared legacy DBAASP-derived 1,423 labels. Exact input file,
  split and training snapshot must be recovered from the SSH artifacts.
- Panel classifier: declared DBAASP-derived `activity_labels_full.csv`, 13,485
  labelled sequences after aggregation. Original snapshot hashes remain needed.
- Hemolysis classifier: declared 4,752 DBAASP-derived labels; recover the actual
  processed filename and original assay-label construction records.
- DRAMP: declared development experiment, not part of the shipped generators.
- ESM backbones: pretrained public components; include model identifiers/revisions
  and their licenses in the final disclosure.

`docs/DATA_DISCLOSURE.md` is a lineage declaration, not proof that each checkpoint
used exactly those files. Do not relicense downloaded public databases under the
repository BSD license. Verify their terms and document access/reconstruction;
any genuinely non-public training data triggers the organizers' separate public
release requirement.

## Run the audit on SSH

```bash
uv run --no-sync python scripts/audit_authorship_readiness.py \
  --out sweep_results/authorship-readiness-v1
cat sweep_results/authorship-readiness-v1/REPORT.md
```

If `gh` is installed/authenticated, optionally add
`--github-repo littlestronomer/amp_challenge_2027` to check visibility/default branch.
Use a new output directory after changes. The audit only inventories files; it
does not upload data, publish the repository, or certify biological performance.

Before public release, recover the three missing predictor config files listed
in the audit and inspect their diff together with checkpoint hashes. Fresh-clone
validation should wait until those files are committed and pushed.

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
