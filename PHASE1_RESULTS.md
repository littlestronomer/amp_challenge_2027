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
| replica (deep training) | 0.626 | 1.075 | 2.921 | 0.741 |

## 10-seed sweep @ fixed 10 epochs (under-trained vs v2)

`bash scripts/sweep_seeds.sh` on SSH, seeds 43–52, GPU-split, v2 architecture,
`data/processed/generative.csv`, **exactly 10 epochs, no early stopping**, generation seed 42.

| metric | mean | std | min | max |
|---|---|---|---|---|
| Uniqueness / Novelty | 1.0 | 0 | 1.0 | 1.0 |
| Diversity | 0.8411 | 0.0020 | 0.8382 | 0.8446 |
| Length | 13.82 | 0.76 | 12.89 | 15.01 |
| FBD | 3.385 | 0.913 | 2.026 | 4.569 |
| MMD | 10.46 | 2.83 | 6.08 | 14.10 |
| Precision | 0.9562 | 0.0139 | 0.9324 | 0.9735 |
| Recall | 0.2964 | 0.1596 | 0.0923 | 0.5467 |
| ConformityScore | 0.7027 | 0.0234 | 0.6493 | 0.7306 |
| Authenticity | 0.7062 | 0.0340 | 0.6521 | 0.7542 |

**Training budget dominates seed.** Monotone trend across budgets (same arch/data):

| budget | Length | Recall | FBD | MMD | Conformity |
|---|---|---|---|---|---|
| 10 epochs (sweep, n=10) | 13.8 ± 0.8 | 0.30 ± 0.16 | 3.38 | 10.5 | 0.70 |
| 59 epochs (replica, seed 43) | 16.7 | 0.74 | 1.07 | 2.9 | 0.63 |
| v2 (unknown budget, ppl 9.58) | 17.4 | 0.83 | 0.51 | 1.1 | 0.59 |

Under-trained models sit at the extreme mode-seeking end: short, charge-heavy,
high-density-core peptides (Conformity up, Recall/FBD/MMD collapsed). v2 must
have trained longer than 10 epochs; historical evidence (hung Aug-9 process)
shows `--epochs 20 --patience 5` style. At fixed budget, most metrics are tight
across seeds (Diversity σ=0.002, Precision σ=0.014) but Recall spreads
0.09–0.55 — coverage is seed-sensitive at short budgets.

## 10-seed sweep @ e100/p10 — the CONFIRMED v2 recipe (seeds 42–51)

`sweep_results/e100_p10/`, seeds 42–51, `--epochs 100 --patience 10`, v2 architecture/data,
generation seed 42. **seed 42 reproduced v2's library metrics exactly** (all displayed digits) —
recipe confirmed, end-to-end determinism re-validated. seed 43 likewise reproduced the earlier
replica (second independent determinism check).

| metric | mean | std | min | max | v2 (seed42) |
|---|---|---|---|---|---|
| Diversity | 0.8443 | 0.0024 | 0.8398 | 0.8483 | 0.8452 |
| Length | 16.74 | 0.74 | 15.28 | 18.00 | 17.36 |
| FBD | 0.906 | 0.467 | **0.192** | 1.888 | 0.513 |
| MMD | 2.44 | 1.58 | **0.553** | 6.100 | 1.113 |
| Precision | 0.919 | 0.025 | 0.854 | 0.945 | 0.909 |
| Recall | 0.764 | 0.091 | 0.574 | **0.913** | 0.834 |
| ConformityScore | 0.603 | 0.042 | 0.497 | 0.653 | 0.591 |
| Authenticity | 0.749 | 0.019 | 0.713 | 0.789 | 0.750 |

**Conclusions:**
1. **v2 was not luck** — it sits within ~1σ of the recipe's seed distribution on every metric
   (mildly favorable on FBD/MMD/Recall, typical elsewhere). All 10 seeds crush the HydrAMP
   baseline.
2. **Seed variance within the recipe is large** (FBD spans 0.19–1.89, a 10× range; Recall
   0.57–0.91; Conformity 0.50–0.65) — seed choice is a real lever, comparable in effect to
   the training-budget and selection-strategy axes.
3. **seed 44 is a coverage champion**: FBD 0.192, MMD 0.553, Recall 0.913, Authenticity 0.789,
   Diversity 0.848, Length 18.0 — beats v2 on all six — at the cost of Precision 0.854 (−0.055)
   and Conformity 0.497 (−0.094). It also beats the *marginal-shaped* v2 library on all four
   of FBD/MMD/Recall/Conformity, making the shaping hack obsolete for the FBD objective.
   Checkpoint: `checkpoint/generator-e100_p10-seed44`, library:
   `generate/submission-e100_p10-seed44/`.

Pareto frontier of candidates (no weighting known): seed44 (distributional coverage) ↔
v2/seed42 (balanced, best Precision among balanced) ↔ seed51 (Precision 0.945 / Conformity
0.653 extreme).

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

## Expanded corpus (MarLys + 3,492 DRAMP net-new) — seed42, confirmed recipe

`checkpoint/generator-e100_p10-expanded`, data `rebuild/generative_expanded_superset.csv`
(42,940 rows = 39,448 canonical + 3,492 DRAMP-strict-new, length μ19.7). Val ppl 6.7
@ epoch 57 early stop (NOT comparable across corpora — val split changed).

| metric | v2 (seed42) | expanded | Δ |
|---|---|---|---|
| Diversity | 0.8452 | **0.8492** | +0.004 |
| Length | 17.36±7.65 | 18.58±8.71 | +1.2 (mechanical) |
| FBD | 0.513 | 0.581 | +0.068 worse |
| MMD | 1.113 | 1.603 | +0.49 worse |
| Precision | 0.909 | 0.904 | −0.005 |
| Recall | 0.834 | 0.844 | +0.010 |
| Conformity | 0.591 | 0.554 | −0.037 worse |
| Authenticity | 0.750 | 0.754 | +0.004 |

**Conclusion: null result.** Every delta is inside the e100/p10 seed-sweep spread
(FBD σ 0.467, Recall σ 0.091, Conformity σ 0.042) except the length shift, which is
mechanically explained by DRAMP's long tail (pool μ30.7). The long-tail hypothesis
(coverage gain without conformity cost) is NOT supported: FBD/MMD/Conformity all
moved the wrong way. Submission lineage stays v2/seed44 until the label side
(classifier retrain on expanded DBAASP labels) shows signal.
Library: `generate/submission-e100_p10-expanded/`; raw clean-rate 69,670/100,000
(vs v2's multi-round flow) is the one mildly positive operational signal.

## Panel-aware ranking adopted (2026-08-29)

Classifier v3 (genus-level, 10 outputs, ESM-2 t12/35M unfrozen-top-4) trained on
the DBAASP-v4 labels: macro AUROC **0.863** (best of 3 seeds; per-genus
0.80–0.89; T=4.6). Label basis: 69,461 DBAASP MIC rows → 39,216 per-genus rows,
13,485 labeled sequences (vs 1,423 legacy) — ~10× the ranker's training signal.

Weight sweep (`sweep_results/panel-weights`, mix m1 = seed44 library — best
coverage: FBD 0.192 / MMD 0.553 / Recall 0.913):

| top-100 recipe | old-binary activity | breadth | mdr |
|---|---|---|---|
| 1,0.5,0.5 (legacy anchor) | 0.938 | 0.365 | 0.317 |
| **0.5,0.25,0.25,1,1 (ADOPTED)** | 0.838 | **1.000** | **1.000** |

Fair-instrument verification (panel classifier probabilities on the two tops):
mean_p 0.593 vs 0.496, mdr_p 0.591 vs 0.493, worst-genus_p **0.515 vs 0.437** —
the adopted top dominates everywhere the 13.5k-label instrument measures; the
anchor's higher "activity" is circular (it was selected by that 1.4k-label head).

Adopted submission configuration: seed44 generator checkpoint + DEFAULT_WEIGHTS
= {activity 0.5, conformity 0.25, precision 0.25, breadth 1, mdr 1}. The entry
point degrades gracefully when classifier_panel.pt is absent (components drop,
weights renormalize).

## De-risking audit (2026-08-30)

**A4.1 — Clustered-split classifier eval (the honest AUROC).** Panel head
retrained with identity-clustered train/val (Levenshtein ratio ≥ 0.70; no
homolog straddles the split; 3 members, best promoted):
**macro AUROC 0.808** (clustered) vs 0.863 (random split) — leakage penalty
−0.055, milder than the −0.10–0.15 typical of AMP datasets. Per-genus under
clustering 0.72–0.83 (E. faecium weakest). All claims now quote 0.808;
the deployed classifier_panel.pt (random-split, more training data) is
unchanged — the clustered run is an evaluation instrument only.

**A4.2 — 650M-embedder robustness check (official fidelity).** Both candidate
libraries re-scored with esm2_t33_650M:

| library | FBD | MMD | Recall | Authenticity | Precision | Conformity |
|---|---|---|---|---|---|---|
| seed44 | **0.269** | **0.570** | **0.906** | **0.782** | 0.851 | 0.497 |
| v2 (seed42) | 0.521 | 0.992 | 0.844 | 0.751 | 0.900 | 0.591 |

seed44's coverage lead survives — and slightly widens — under the
official-fidelity embedder; v2 keeps its Precision/Conformity edge exactly as
at 8M. Instrument-bet de-risked: **seed44 remains the submission library.**

**A4.3 — hemolysis data audit (in progress).** Refetched with the corrected
v2 schema (13,892 hemolytic peptides; server-side filter). The data is
(concentration, percent-lysis-band) pairs on erythrocytes — IC50 rows are
non-RBC cytotoxicity and are excluded by the target filter. 9,010 sequences
carry usable observations; label rule risk-band 40% / safe-band 30% /
ceiling 128 µM validated by the full kind distribution.

**A4.3 — hemolysis safety head (closed 2026-08-30).** Labels: 4,752 sequences
(2,568 risky / 2,184 safe) from the band rule (risk ≥40% lysis at ≤128 µM;
safe ≤30% at ≥128 µM; erythrocyte rows only). Head: promoted member AUROC
**0.932** (T=3.94), artifacts in `checkpoint/reward_hemo/` (separate dir).
Audit of the adopted top-100: p(risky) mean 0.382 / p75 0.395 / **max 0.434**
— no concentrated risk ⇒ the adopted recipe is UNCHANGED (`--w-safety`
defaults to 0; the component is wired and available). Caveats: melittin
sanity passed in direction (0.495 vs control 0.388) with modest absolute
separation under T=3.94 — treat p_risky as a conservative ranking signal;
AUROC 0.932 is random-split (informational only while weight is 0).

**De-risk phase COMPLETE — position verified.** Honest AUROC 0.808 (≥0.75
gate), library choice confirmed under the 650M embedder, no safety exposure.
Improvement tracks may resume from this baseline.

## L0 decode-parameter sweep (2026-08-30) — top-p 0.95 ADOPTED

3×3 grid around the hand-set default (T/top-p/rep-pen) on the seed44
checkpoint (`sweep_results/decode/summary.tsv`). Anchor reproduced the
recorded metrics exactly (driver + determinism verified). Monotone gradient:
hotter/looser decoding improves coverage; the sharp corner collapses
(FBD 0.418). Adopted cell — **top-p 0.9 → 0.95** (single-knob change):

| | anchor | adopted | Δ |
|---|---|---|---|
| FBD | 0.192 | **0.148** | −23% |
| MMD | 0.553 | **0.221** | −60% |
| Recall | 0.913 | 0.921 | +0.008 |
| Conformity | 0.4966 | 0.4965 | flat |
| Precision | 0.854 | 0.850 | −0.005 |

Runner-up (aggressive alt): T=1.15/top-p=0.95/rep=1.2 — best Recall 0.930 and
Conformity +0.006 at FBD 0.176. Cross-seed σ is large but measures checkpoint
variation; for a fixed checkpoint this comparison is deterministic, consistent
across 4 cells, and embedder-stable per A4.2. 650M confirmation + top-100
potency re-verification follow before the swap is final.

**L0 AMENDED (2026-08-30): top-p 0.95 REVERTED — 8M-embedder artifact.**
The 650M confirmation failed: FBD 0.274 vs 0.269 and MMD 0.600 vs 0.570 at
top-p 0.95 (vs 0.9) — the 8M-only "−23% FBD / −60% MMD" did not transfer to
the official-fidelity embedder. Small counter-gains (Recall +0.008, Diversity
+0.004, Authenticity +0.008) are not decision-grade. Default reverted to 0.9;
the submission library remains the original seed44 decode.

**Lesson (binding for future library changes):** 8M-embedder FBD deltas below
~0.05 (and MMD below ~0.3) are NOT decision-grade — the 650M embedder is the
gate instrument for any library-affecting change. Coarse rankings (seed44 vs
v2) transferred; fine decode deltas did not. L0 closed as null: decode
parameters are not a reliable lever at this delta scale.

## seqme benchmark tutorial cross-check (2026-08-30)

The szczurek-lab seqme tutorial table is NOT our protocol: 3,000-sample
subsets, three reference datasets (UniProt/AMPs/AMP-data) with per-metric
references, conformity on [amphiphilicity, charge], plus FKEA and amPEPpy
which we did not compute. No row-to-row comparison with our numbers is valid.
Actions taken: FKEA and a tutorial-configured twin ConformityScore added to
metrics_official (informational); amPEPpy flagged as the one missing axis —
it needs a py3.8 conda env (seqme-thirdparty) and is follow-up infra. If the
official Phase-1 suite includes an activity surrogate, our panel-classifier
breadth is a proxy, not a guarantee, of that axis.

## G1 charge-conditioned SFT (2026-08-30) — mechanism success; kept as blend pool, NOT a replacement

`checkpoint/generator-e100_p10-cond` (canonical corpus, v2 recipe +
`--conditioning charge`, early stop ep55, val ppl 6.4). **Charge marginal
fixed as designed: pool σ 3.15 vs reference 3.28** (unconditioned band was
2.4–2.9); other marginals undamaged (hmoment exact, hydro/length close);
clean-candidate rate highest yet (~80%).

| 650M gate | seed44 | cond |
|---|---|---|
| FBD / MMD / Recall | **0.269 / 0.570 / 0.906** | 0.493 / 1.063 / 0.830 |
| Precision / Conformity | 0.851 / 0.497 | **0.897 / 0.535** |
| top-100 (breadth/mdr/mean_p) | 1.0/1.0/0.594 | 1.0/1.0/0.589 |

The tradeoff transferred across embedders (property-space conformity is
instrument-independent; FBD direction consistent at 8M and 650M — unlike the
L0 decode artifact). Verdict: NOT a seed44 replacement (coverage gate failed);
retained as the second pool for hybrid-library composition. FKEA baselines
captured: 519.4 (8M) / 1788.9 (650M) for the cond library.

## Hybrid library (seed44 + cond, 50/50 interleave) — BEST LIBRARY RESULT (2026-08-30)

Deterministic round-robin interleave of the two 50k libraries (25,202/25,197
after dedup), evaluated at the 650M gate:

| 650M | seed44 | cond | hybrid |
|---|---|---|---|
| FBD | 0.269 | 0.493 | **0.236** |
| MMD | 0.570 | 1.063 | **0.381** |
| Precision | 0.851 | 0.897 | 0.876 |
| Conformity | 0.497 | 0.535 | 0.515 |
| Recall | 0.906 | 0.830 | 0.889 |

The mixture beats BOTH components on FBD/MMD (distributions bracket the
reference from different sides) and takes the better side of both parents on
Precision/Conformity; sole cost Recall −0.017 vs seed44. Blend ratio is an
open knob (50/50 was the first probe). Adoption requires wiring multi-
checkpoint blending into the entry point with byte-determinism — not yet
implemented; submission stack unchanged until then.
