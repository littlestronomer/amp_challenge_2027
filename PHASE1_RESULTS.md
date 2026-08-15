# Phase-1 performance record

## Libraries on SSH machine (`istke-compute-1`, GPU 1, RTX 5090 ×2)

### Baseline v2 — `generate/submission-v2/`
Generator: `checkpoint/generator-attnres-v2`, `--repetition-penalty 1.3`, 100000 raw → 50000 clean (1 round). Val ppl 9.58.

| metric | value | deviation |
|---|---|---|
| Uniqueness | 1.0 | — |
| Novelty | 1.0 | — |
| Diversity | 0.845167 | — |
| Length | 17.36034 | 7.651479 |
| NGramJaccard | 0.00189 | — |
| FBD | 0.513002 | — |
| MMD | 1.113176 | — |
| Precision | 0.90914 | — |
| Recall | 0.834161 | — |
| ConformityScore | 0.590889 | 0.001193 |
| Authenticity | 0.74982 | — |

### Marginal-shaped — `generate/submission-shaped/`
Same checkpoint/penalty; `--pool-multiplier 3`, 255000 raw → 164215 clean (2 rounds), property-marginal bucket-filling.

| metric | value | deviation | Δ vs v2 |
|---|---|---|---|
| Uniqueness | 1.0 | — | — |
| Novelty | 1.0 | — | — |
| Diversity | 0.846264 | — | +0.001 |
| Length | 17.96966 | 8.154453 | +0.61 |
| NGramJaccard | 0.001959 | — | +0.00007 |
| FBD | 0.368947 | — | −0.144 |
| MMD | 0.712633 | — | −0.401 |
| Precision | 0.90814 | — | −0.001 |
| Recall | 0.835404 | — | +0.001 |
| ConformityScore | 0.542976 | 0.001105 | −0.048 |
| Authenticity | 0.7502 | — | +0.0004 |

### Converged replica (seed 43, 100-epoch budget, early stop epoch 59) — `generate/submission-v2-replica/`
Val ppl 7.02. Not a strict v2 replication — trained ~6× longer than v2 evidently did.

| metric | value | deviation | Δ vs v2 |
|---|---|---|---|
| Uniqueness | 1.0 | — | — |
| Novelty | 1.0 | — | — |
| Diversity | 0.843443 | — | −0.002 |
| Length | 16.66346 | 7.005058 | −0.70 |
| NGramJaccard | 0.001818 | — | −0.00007 |
| FBD | 1.074919 | — | +0.562 |
| MMD | 2.920866 | — | +1.808 |
| Precision | 0.9336 | — | +0.024 |
| Recall | 0.741153 | — | −0.093 |
| ConformityScore | 0.626048 | 0.00118 | +0.035 |
| Authenticity | 0.73932 | — | −0.010 |

### Tradeoff summary
| library | Conformity | FBD | MMD | Recall |
|---|---|---|---|---|
| marginal-shaped (spread selection) | 0.543 | **0.369** | **0.713** | 0.835 |
| v2 (moderate training) | 0.591 | 0.513 | 1.113 | **0.834** |
| replica (deep training) | **0.626** | 1.075 | 2.921 | 0.741 |

Reference: `data/antibacterial.fasta`, 39448 sequences. Embedder for eval: `facebook/esm2_t6_8M_UR50D`.

## Property distributions (props.py scale)

| set | charge | hydro (KD) | hmoment |
|---|---|---|---|
| reference (full) | +2.88 ± 3.28 | −0.29 ± 1.01 | +0.280 ± 0.178 |
| v2 pool | +2.65 ± 2.43 | −0.07 ± 0.91 | +0.296 ± 0.175 |
| marginal-shaped | +2.81 ± 2.95 | −0.20 ± 0.94 | +0.288 ± 0.175 |

seqme (modlamp) reference: charge +2.85 ± 3.28, hydro −0.067 ± 0.381, hmoment +0.395 ± 0.199.

## ConformityScore definition (`seqme/metrics/conformity_score.py`)
Gaussian KDE (bandwidth `"silverman"`, 5-fold KFold, seed 0) over reference joint property vectors. Score per generated peptide = fraction of reference val points with log-density ≤ its log-density; mean over peptides and folds. Objective: maximize. Predictors: seqme `Charge`, `Hydrophobicity`, `HydrophobicMoment` (require `seqme[aa_descriptors]` → `modlamp`).

## Files (branch `feat/property-distribution-shaping`, pushed to origin)
- `src/amp_challenge_2027/shape.py` — `shape_library_to_reference` (marginal bucket-filling, pure numpy).
- `scripts/generate_shaped.py` — entry point; reuses `generate.py`/`select.py` via import.
- `tests/test_shape.py` — 9 tests, pass.
- `scripts/sweep_seeds.sh` — 10-seed train→generate→evaluate sweep (default 10-epoch budget, seeds 43–52).
- `generate.py`, `select.py` — untouched on this branch (charge-conditioning work lives on `feat/charge-conditioning`).

## Machine setup
- Dev: `tinystronomer`, RTX 5070 Ti.
- SSH/training: `istke-compute-1`, 2× RTX 5090; commands use `CUDA_VISIBLE_DEVICES=1`, `uv run --extra ml`.
- SSH repo: `~/Documents/Project/amp_challenge_2027`; pulls from fork via HTTPS (username/password).
- Classifier present on SSH: `checkpoint/reward/classifier.pt` + `config.json` (`facebook/esm2_t12_35M_UR50D`). Not present on dev machine.
- eval invocation on SSH uses importlib pattern (direct `scripts/eval_official.py` produced no output).
- Original v2 training invocation (user-reconstructed, uncertain): `--epochs 100 --patience 10` produced ppl 7.02 at seed 43, so the actual v2 budget was likely shorter (~10 epochs) to yield 9.58.
