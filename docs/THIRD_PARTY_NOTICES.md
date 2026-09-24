# Third-party sources and acknowledgements

Reviewed against the linked primary sources on 2026-09-25. This records their
published terms, not a certification of patent clearance or of every historical
download's terms. The repository's BSD-3-Clause license does not replace source
data terms. Existing upstream copyright notices remain in LICENSE and Git history.

| Source | Role and attribution | Published terms / evidence |
|---|---|---|
| AMP Challenge starter | Repository scaffold, competition reference corpus and validation interface; Ewa Szczurek lab and starter contributors | [Official starter](https://github.com/szczurek-lab/amp-challenge-2027), BSD-3-Clause |
| MarLys / MLAMP_db | Declared generator corpus lineage; Marczak, Bocian and Łyskowski, version 3, DOI 10.17632/w4hb5grjwb.3 | [Mendeley version 3](https://data.mendeley.com/datasets/w4hb5grjwb/3) lists CC0 1.0. The exact competition subset and CSV still require source reconciliation. |
| DBAASP | Activity, panel and hemolysis training labels are declared DBAASP-derived. We acknowledge DBAASP, its data contributors, IBCEB and NIAID/OCICB. Cite Pirtskhalava et al., DBAASP v3, NAR 49(D1):D288–D297 (2021), DOI 10.1093/nar/gkaa991. | [Current data-use policy](https://dbaasp.org/terms-and-conditions) permits access, adaptation and redistribution with public acknowledgement, including for modified/combined data. Raw snapshots are not included in this release preparation. |
| DRAMP | Separately downloaded development data; also one of the upstream databases listed by MarLys. Cite Shi et al., DRAMP 3.0, DOI 10.1093/nar/gkab651 for the 3.0 resource. | [DRAMP](https://dramp.cpu-bioinfor.org/) states CC BY 4.0, requests original-article citations and describes authorization for patent AMPs. [DRAMP 3.0 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8728287/). Download URL folder names alone do not pin a database release. |
| ESM-2 | Meta pretrained protein language models: 35M classifier backbones, 8M precision proxy, 650M historical evaluation | [35M model card](https://huggingface.co/facebook/esm2_t12_35M_UR50D) and [upstream license](https://github.com/facebookresearch/esm/blob/main/LICENSE) identify MIT. We download these pretrained components; their pretraining data are external to our own training snapshots. Historical exact backbone revisions remain unverified. |

The MarLys record names AMPDB, dbAMP, DRAMP, CAMP, DBAASP, SATPdb, APD, CyBase,
InverPep, DADP, CancerPPD, BaAMPs and ParaPep as its upstream databases. The author
supplied an SSH report confirming that all 13 names occur in `source_dbs` for the
current 39,448-sequence CSV. Counts overlap across databases and do not independently
verify each source attribution. See [the recorded report](release/SSH_LINEAGE_REPORT.json).

The old downloader's citation string conflates DRAMP 2.0 (2019) with DRAMP 3.0.
Historical manifests remain unchanged for integrity; the citation is corrected
here. Rights to conduct wet-lab or commercial work are not established by a
database's access license or by sequence novelty against one reference set.
