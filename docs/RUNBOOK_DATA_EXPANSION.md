# Runbook — expanding the generative training corpus

Goal: grow SFT data beyond MarLys (~102k) so Recall/FBD (coverage metrics) stop
being capped by training diversity. Two zero-fetch wins + optional manual
downloads; everything lands in one merged corpus.

## What gets ingested

| source | how it arrives | where it's read from |
|---|---|---|
| MarLys (existing) | already processed | `data/processed/generative.csv` |
| DBAASP sequences | **already on disk** — extracted from `mic.csv`/`peptides.csv` | automatic |
| DRAMP / APD / dbAMP / LAMP | manual download (click-mediated pages; no stable direct URLs) | drop FASTA under `data/raw/<source>/` |

Curation applied to everything: standard 20 AAs only, length 8–50, global
exact-dedup (MarLys > DRAMP > APD > … priority), and exact matches against
`data/antibacterial.fasta` are dropped so the model never memorizes a sequence
the submission must avoid.

## Steps

1. (optional) Fetch extra sources — the fetcher prints exact instructions:

   ```bash
   uv run --extra ml python scripts/fetch_data.py --source dramp
   # open https://dramp.cpu-bioinfor.org/downloads/, download the general
   # dataset (FASTA or zip), then:
   uv run --extra ml python scripts/fetch_data.py --source dramp \
       --url '<direct-file-url>'     # or place the file at data/raw/dramp/ yourself
   ```

   Repeat for `--source apd` / `--source dbamp` if desired.

### 2a. One-command wrapper (recommended — keeps the canonical corpus frozen)

`scripts/rebuild_corpus.py` chains steps 2–3's data side into a single
deterministic command, but writes everything to a **separate path**
(`data/processed/rebuild/`) and hash-verifies that no pre-existing file under
`data/processed/` changed during the run (exit code 3 + explicit list if it
did). `data/processed/generative.csv` — the corpus the seed-44 checkpoint was
trained on — is therefore safe: reproduction of the confirmed v2/seed-44
recipe stays byte-exact no matter how often you rebuild.

```bash
uv run python scripts/rebuild_corpus.py
```

What it does:

1. Snapshots SHA-256 of every file under `data/processed/`.
2. Rebuilds the base curated sets from raw inputs via the same parse+curate
   code as the original build, into the sandbox, then reports whether the
   rebuilt `generative.csv` is byte-identical to the canonical one (a free
   upstream-data/curation drift check).
3. Merges base + raw FASTAs + DBAASP sequences into
   `data/processed/rebuild/generative_expanded.csv`, reusing
   `scripts/build_expanded_generative.py`'s loaders so merge rules cannot
   drift.
4. Re-snapshots; any pre-existing file that changed aborts with exit code 3.

Outputs never collide with canonical artifacts. Train on the expanded corpus
with `--data data/processed/rebuild/generative_expanded.csv` (step 3 below).
Manual steps 2–3 remain valid if you prefer them; use only one path per run.

2. Build the merged corpus:

   ```bash
   uv run --extra ml python scripts/build_expanded_generative.py
   ```

   Prints per-source counts, dedup stats, and charge/KD/length summaries.
   Writes `data/processed/generative_expanded.csv`.

3. Retrain on it (v2 recipe; combine with conditioning if running both tracks):

   ```bash
   CUDA_VISIBLE_DEVICES=1 uv run --extra ml python scripts/train_generator.py sft \
       --data data/processed/generative_expanded.csv \
       --residual block_attnres --epochs 100 --patience 10 --precision bf16 \
       --out-dir checkpoint/generator-e100_p10-expanded --log-dir runs/e100_p10-expanded
   ```

4. Generate a pool and add it to `sweep_selection.py --pools ...` exactly like
   any other checkpoint (see RUNBOOK_SELECTION_SWEEP.md).

## Expectations

- DBAASP alone typically adds thousands of unique peptides with zero new
  downloads (the script reports the exact number).
- DRAMP general is ~6k entries; patent entries are excluded by design
  (licensing) unless your collaborator decides otherwise.
- If `extra == 0`, the script says so — you likely have no FASTAs under
  `data/raw/` yet.

## Expanded ranking-label datasets (the two sources that matter)

Two feeds make the *ranking* side stronger, independent of generative-corpus size:

1. **DBAASP, fully exploited** — the legacy binarizer keeps ~2k rows; the
   per-genus builder multiplies usable labels at zero fetch cost:

   ```bash
   uv run python scripts/build_ranking_labels.py   # reads data/processed/mic.csv
   ```

   | output | shape | feeds |
   |---|---|---|
   | `data/processed/activity_labels_full.csv` | one row per (sequence × panel genus): min MIC µM, band (`potent/active-band/weak/inactive`), binary label vs `MIC_SUCCESS_THRESHOLD_UM` (blank = masked) | activity-classifier retrains |
   | `data/processed/activity_labels_gram.csv` | per-sequence Gram-type votes, DBAASP numeric evidence taking precedence over DRAMP membership (`source` column tells you which) | broad-spectrum / future panel-aware ranking |

   µg/mL rows are converted via deterministic peptide MW instead of being
   misread as µM; unsupported units are dropped and counted in the printed
   aggregation stats.

2. **DRAMP 3.0 splits (CC BY 4.0)** — direct downloads with provenance recorded
   to `data/raw/sources.json` ({url, retrieved_utc, license, sha256}) which doubles
   as the training-data disclosure skeleton:

   ```bash
   uv run python scripts/fetch_data.py --source dramp-general       # generative pool
   uv run python scripts/fetch_data.py --source dramp-antibacterial # generative pool
   uv run python scripts/fetch_data.py --source dramp-grampos       # label split (probe)
   uv run python scripts/fetch_data.py --source dramp-gramneg       # label split (probe)
   ```

   The Gram-split anchors are malformed on DRAMP's downloads page today, so the
   last two probe plausible paths and fall back to exact manual instructions on
   failure — check output before assuming success. FASTAs land under
   `data/raw/dramp/`; rerun `scripts/rebuild_corpus.py` afterwards so they flow
   into `generative_expanded.csv`, and rerun `build_ranking_labels.py` so split
   memberships enter the Gram labels.

Everything here writes NEW files; canonical artifacts (incl. seed-44's
`generative.csv`) stay frozen by construction.

## Licensing note (full-track requirement)

Training-data disclosure must cover every source kept in the expanded corpus.
DRAMP is CC BY 4.0; check APD/dbAMP terms before including them in the public
repo artifacts. If a source can't be redistributed, keep it out of the corpus
rather than shipping an undisclosed derivative.
