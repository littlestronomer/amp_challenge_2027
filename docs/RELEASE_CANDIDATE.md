# Release candidate: littlestronomer

Author: Göktürk Batın Dervişoğlu. Correspondence: dervisoglu21@itu.edu.tr.
Affiliation remains awaiting the author's answer; no institution is inferred
from the email domain. Starter contributors retain their credit and copyright;
Git commit authorship alone is not a proposed scientific author list.

## Validated inference baseline

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

GitHub was checked on 2026-09-25: repository private; default branch `main`.
The release candidate is the existing feature branch, proposed for review into
`main`. Until that promotion, ordinary default-branch clones do not identify
this validated candidate. Use the explicit branch or baseline commit for review.

1. SSH collection is complete as reported; retain the historical-link limitations
   unless contemporaneous training records are recovered.
2. Supply author affiliation and review source attribution and release contents.
3. Review the release PR before promoting the candidate to the default branch.
4. Publish the reviewed repository and verify unauthenticated access to the
   exact release branch/weights. Visibility has not been changed by preparation.
5. Submit the hash-matched FASTAs and release reference, then retain the official
   receipt. No competition submission or organizer message is sent by these tools.

Public access and full data disclosure are among the organizers'
[full co-authorship requirements](https://github.com/szczurek-lab/amp-challenge-2027#submission-requirements).
The unresolved historical data links are disclosed rather than certified away.
Organizers determine eligibility; passing a local validator is not admission.
