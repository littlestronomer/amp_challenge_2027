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

## Licensing note (full-track requirement)

Training-data disclosure must cover every source kept in the expanded corpus.
DRAMP is CC BY 4.0; check APD/dbAMP terms before including them in the public
repo artifacts. If a source can't be redistributed, keep it out of the corpus
rather than shipping an undisclosed derivative.
