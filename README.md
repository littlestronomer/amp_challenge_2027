# AMP Challenge 2027

> International competition for generative AI in antimicrobial peptide design.

**Submission: littlestronomer — Göktürk Batın Dervişoğlu**
— Istanbul Technical University
([dervisoglu21@itu.edu.tr](mailto:dervisoglu21@itu.edu.tr)).
See the [method abstract](docs/ABSTRACT.md),
[training-data disclosure](docs/DATA_DISCLOSURE.md), and
[authorship release audit](docs/AUTHORSHIP_READINESS.md).
The [release candidate](docs/RELEASE_CANDIDATE.md) records the validated commit
and FASTA hashes. See [training provenance](docs/TRAINING_PROVENANCE.md) for
snapshot evidence and unresolved links, and [third-party notices](docs/THIRD_PARTY_NOTICES.md)
for source attribution. Citation metadata is in [CITATION.cff](CITATION.cff).
The [repaired selector runbook](docs/RUNBOOK_CONSTRAINED_SELECTIVITY.md)
describes the frozen-pool experiment and invalidates earlier faulty solver results.

Antimicrobial resistance is one of the most pressing global health challenges. This competition invites participants to develop generative models that design novel antimicrobial peptides (AMPs) with activity against a panel of clinically relevant bacterial strains, including multi-drug resistant ESKAPE pathogens.

## Submission Requirements

### Minimum (benchmark participation)
- Abstract summarizing the method
- Library of 50,000 designed AMPs
- Ranked top-100 candidates with selection/ranking documentation
- Short summary of training data, external databases, and any filters applied
- GitHub repository (private is fine) with model weights and inference code; grant read access to [@RasmusML](https://github.com/RasmusML) and [@szymczakpau](https://github.com/szymczakpau)

### Full (co-authorship eligibility)
All of the above, plus:
- Public GitHub repository with model weights, inference code, and usage docs
- Permissive OSI-approved license (MIT, BSD-3-Clause, or Apache 2.0)
- Uses **[`uv`](https://docs.astral.sh/uv/concepts/projects/init/#projects)** for dependency management (include `uv.lock` and a defined Python version)
- Entry point runnable via `uv run generate` generating the 50,000-member library and top-100 list; any additional arguments must have defaults
- Fixed default random seed (identical output on repeated runs)
- Full training data disclosure; any non-public data must be released under a permissive license

## Sequence Requirements

Generated sequences must:

- Use only the 20 standard proteinogenic amino acids (`ACDEFGHIKLMNPQRSTVWY`)
- Be between 8 and 50 residues long
- Be unique (no duplicates)
- Be linear with free termini (no terminal modifications, including amidation)
- Exclude noncanonical amino acids, stapled peptides, peptidomimetics, and chemically modified variants (lipidated, glycosylated, PEGylated, dendrimeric, etc.)

The full 50,000-sequence library must additionally contain no sequences identical to known antibacterial peptides in `data/antibacterial.fasta`. The top-100 list is held to a stricter standard: no sequence may exceed 80% sequence identity (Levenshtein ratio) with any sequence in that reference set.

## Getting Started

To run this trained submission, clone the public default branch (`main`) and run
its default entry point:

```bash
git clone https://github.com/littlestronomer/amp_challenge_2027.git
cd amp_challenge_2027
uv sync
uv run generate
```

Outputs are `generate/library.fasta` and `generate/top.fasta`. The author validated
revision `7a5b86bf11f43a207a2a2c7f2b9c0fd82c22169d` on 30 September 2026:
two default runs and a strict run produced identical FASTAs matching the recorded
submission hashes. `main` contains that implementation plus documentation updates;
the newer research branch is not the submitted implementation. To reproduce the
exact tested checkout, run `git checkout 7a5b86bf11f43a207a2a2c7f2b9c0fd82c22169d`
before `uv sync`. See the [validation record](docs/release/VALIDATION_2026-09-30.json)
and [release notes](docs/RELEASE_CANDIDATE.md) for evidence and limitations.
The SSH run used physical GPU 1 via `CUDA_VISIBLE_DEVICES=1`; choose a suitable
available GPU for your machine. Identical results across all environments are not
guaranteed.

The following starter-template instructions are retained for developing a
separate submission, rather than running these supplied weights.

This repository also serves as a working example — see [src/amp_challenge_2027/generate.py](src/amp_challenge_2027/generate.py) for a complete implementation that meets all requirements.

The steps below walk through building a minimal submission. Replace `my-model` with your model name throughout.

### 1. Initialize the project

```bash
uv init --package my-model
cd my-model
```

### 2. Add the entry point

In `pyproject.toml`, add a `[project.scripts]` section:

```toml
[project.scripts]
generate = "my_model.generate:main"
```

Note: to add package dependencies, use `uv add <package>` instead of editing `pyproject.toml` directly.

### 3. Implement `generate.py`

Running the entry point produces two files in a `generate/` subdirectory:

```
generate/
  library.fasta  ← full 50,000-sequence library
  top.fasta      ← top-100 ranked sequences
```


See [src/amp_challenge_2027/generate.py](src/amp_challenge_2027/generate.py) for a complete example.

### 4. Run locally

Install dependencies and test your script:

```bash
uv run generate
```

> **Note:** the base install includes `torch` + `transformers` on purpose —
> the competition validator runs plain `uv run generate`, and that must
> execute the trained model rather than a fallback sampler. Training-only
> dependencies stay optional: `uv sync --extra ml` (plus `--extra seqme` for
> local protocol scoring).

Optional arguments (must have defaults):

| Flag | Default | Description |
|------|---------|-------------|
| `--n-sequences` | `50000` | Number of sequences to generate |
| `--top-k` | `100` | Number of top-ranked sequences to write |
| `--seed` | `42` | Random seed for reproducibility |
| `--length` | `50` | Length of each generated sequence |

### 5. Verify

Push your project (including `uv.lock`) to a **public** GitHub repository, then run the validator:

```bash
uv run python scripts/verify_submission.py <github-url>
```


### 6. Submit

To submit, head to the Kaggle competition page: https://www.kaggle.com/competitions/amp-challenge

## Validation

For controlled generator checkpoint and 50k-library selection experiments on
the SSH training machine, see [the generator experiment runbook](docs/RUNBOOK_GENERATOR_EXPERIMENTS.md).
For fixed-ratio epoch-58 + charge-conditioned blends using saved pools, see
[the blend-ratio runbook](docs/RUNBOOK_BLEND_RATIOS.md).
For the paired top-100 audit after blend confirmation, see
[the top-100 comparison runbook](docs/RUNBOOK_TOP100_COMPARISON.md).
For cached-score diagnosis, fixed-normalization ablation and predictor artifact
inventory, see [the selection diagnosis runbook](docs/RUNBOOK_SELECTION_DIAGNOSIS.md).
After matching the frozen predictor members, see
[the provisional reconstruction runbook](docs/RUNBOOK_REWARD_RECONSTRUCTION.md)
for hash-pinned validation diagnostics with explicit historical-provenance limitations.
For fresh predictor training with joint sequence-family train/validation/calibration/test
splits, linear controls and a sealed final test, see
[the generalization benchmark runbook](docs/RUNBOOK_REWARD_GENERALIZATION.md).
For the strict incumbent release check, hash-linked result inventory and whole-top-100
readiness audit, see [the competition readiness runbook](docs/RUNBOOK_COMPETITION_READINESS.md).
The current claim status and next-experiment gate are summarized in
[the competition status page](docs/COMPETITION_STATUS.md) and
[the claims register](docs/CLAIMS_REGISTER.md).

Verify your submission with:

```bash
uv run python scripts/verify_submission.py <github-url>
```

This clones your repo, installs dependencies, generates the full library and ranked top-100 into `generate/library.fasta` and `generate/top.fasta`, verifies both files, then generates them again to confirm the output is reproducible.

| Argument | Default | Description |
|----------|---------|-------------|
| `url` | — | GitHub repository URL (required positional) |
| `--branch` | repo default | Git branch to clone |
| `--dir` | `submission/` | Directory to clone into |
| `--extra` | — | Optional [uv](https://docs.astral.sh/uv/concepts/projects/init/#projects) extras to install (repeatable) |
| `--antibacterial-fasta` | `data/antibacterial.fasta` | FASTA file of known antibacterial sequences to check for overlap |

## Starter Kits

The following starter kits are compatible with this submission format:

- [ampdiffusion-starter-kit](https://github.com/szczurek-lab/ampdiffusion-starter-kit)
- [hydramp-starter-kit](https://github.com/szczurek-lab/hydramp-starter-kit)

## Project Structure

```
amp-challenge-2027/
├── checkpoint/
│   └── weights.csv          # Trained model weights
├── scripts/
│   └── verify_submission.py # Submission validator
├── src/
│   └── amp_challenge_2027/
│       └── generate.py      # Entry point: sequence generation logic
├── pyproject.toml
└── uv.lock
```
