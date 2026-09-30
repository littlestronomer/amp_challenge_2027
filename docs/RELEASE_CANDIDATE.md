# Release candidate: littlestronomer

Author: Göktürk Batın Dervişoğlu. Correspondence: dervisoglu21@itu.edu.tr.
Affiliation: Istanbul Technical University, supplied by the author on 2026-09-25.
Starter contributors retain their credit and copyright;
Git commit authorship alone is not a proposed scientific author list.

## Validated submission revision

Revision `7a5b86bf11f43a207a2a2c7f2b9c0fd82c22169d` passed the author's SSH
fresh-clone validator on 2026-09-30. Two default runs passed sequence constraints,
reference exclusion/similarity and byte equality. A third, strict generation run
required all five intended scoring components and produced identical FASTAs.
Both output hashes match the historical baseline below. Downloaded FASTAs,
logs, manifest and audit were inspected locally; no GPU inference was repeated
on the manuscript workstation. The compact [validation record](release/VALIDATION_2026-09-30.json)
records attribution, artifact hashes and audit findings.

`main` now contains this tested implementation and documentation updates. These
updates do not modify inference code, weights, reference data, dependencies or
default settings. The exact tested revision remains available by its full hash.

## Historical inference baseline

Commit `be3aecae7cdde9bc898337c3b282d3fa4714457f` on
`feat/nway-blend-multiaxis-conditioning` passed the SSH fresh-clone default
validator, including two-run byte equality, according to author-supplied output
on 2026-09-25. SHA-256 comparison also matched the frozen competition-readiness
library and top 100. [BASELINE.json](release/BASELINE.json) preserves the exact
hashes, evidence attribution and committed inference-file hashes.

| Output | SHA-256 |
|---|---|
| `library.fasta` | `74f73e71e6d867083aede2cd773ec4761e0f2533edf7485416447d8023c4b84b` |
| `top.fasta` | `eea9528d64e3d125f3639772239ccfc39fed25fb3746100670f0e27b58c08221` |

This release-preparation update changes documentation and CPU evidence tooling;
inference code, weights, dependency lock and default settings are kept at the
validated baseline. File-hash equivalence is checked by the lineage collector.
It does not guarantee output portability to every hardware/runtime combination.

## Release contents

- Model weights, dedicated inference configs, reference FASTA, inference source,
  dependency lock, Python version, tests and validation tools.
- [Method abstract](ABSTRACT.md), [data disclosure](DATA_DISCLOSURE.md),
  [training lineage](TRAINING_PROVENANCE.md), [third-party notices](THIRD_PARTY_NOTICES.md),
  [author identity](SUBMISSION_IDENTITY.json) and [citation metadata](../CITATION.cff).
- Historical experiments and their corrections. The hemolysis selector has not
  demonstrated an improvement; no wet-lab activity or safety claims are made.
- Raw/processed training files, generated libraries, experiment caches and local
  credentials are not intentionally added by this preparation. Existing Git
  history remains intact; publication exposes historical commits as well.

The [content review record](release/CONTENT_REVIEW.json) covers 472 text blobs
across locally available Git refs, using common credential/key patterns; no
matches were found. Eight large/binary blobs were excluded. This is a bounded
review, not comprehensive secret detection or an audit of serialized weights.
Git author identities remain in history. Existing starter notices are preserved.

## Branch and publication sequence

The author-supplied [SSH lineage report](release/SSH_LINEAGE_REPORT.json) at
`928e7fd41d035d9950861b37175d84405a49519a` matches all deployed weights and candidate
CSVs, identifies original-run model copies, and confirms the baseline runtime and
FASTA hashes. The current generator CSV contains exactly the reference sequence
set and annotations for all 13 upstream databases. Original training-time data
and split linkage remains unverified and is explicitly disclosed.

Authenticated GitHub checks on 2026-09-30 confirmed public visibility and `main`
as the default branch after promotion of the validated revision. The author
reported completing the Kaggle submission on the same date; the submitted entry
and receipt have not been independently inspected.

Remaining actions:

1. Confirm that the public-source inventory is exhaustive and reconcile any
   original training-input/split evidence that can be recovered. Inference
   validation does not establish this provenance.
2. Retain the Kaggle receipt and verify that the submitted FASTAs and revision
   match the hashes above. Preserve the validation archive.
3. Share the public repository link through the competition's Discussion forum
   or a competition notebook, as required by the public-code-sharing provision
   in the [Kaggle rules](https://www.kaggle.com/competitions/amp-challenge/rules).
   No such post has been made by this documentation update.
4. Respond to any organizer compliance requests. No further generation is
   needed solely for these documentation corrections; changes to runtime files
   require a new validation assessment.

Public access and full data disclosure are among the organizers'
[full co-authorship requirements](https://github.com/szczurek-lab/amp-challenge-2027#submission-requirements).
The unresolved historical data links are disclosed rather than certified away.
Organizers determine eligibility; passing a local validator is not admission.
