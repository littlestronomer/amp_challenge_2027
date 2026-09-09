# Molar-only candidate dataset and retention gate

On SSH, from the repository root:

```bash
git pull --ff-only origin main
uv run --no-sync python scripts/build_molar_candidates.py \
  --source sweep_results/label-observations-v1 \
  --prepared sweep_results/reward-generalization-data-v1 \
  --out sweep_results/molar-candidates-v1
```

CPU only. Requires the completed observation audit and existing prepared family
partition, verified through completion hashes. No input, existing split, model,
or label file is overwritten. Use a new output directory.

For each task, outputs include:

- `*_candidates.csv`: sequence, organism, label; diagnostic candidate labels only.
- `*_groups.csv`: retained/masked groups, source lines and exact target strings.
- `*_observation_decisions.csv`: every observation's selection/exclusion reason.
- `*_population.csv`: candidate labels alongside old-label availability and the
  existing family/split membership. Unmapped sequences stay unmapped.
- `report.json`: class counts, unique sequences, group decisions, common old-label
  keys and family/split coverage. Empty tables are omitted and counts remain zero.

Only standard 8–50-residue sequences, task-eligible targets, and scalar direct
molar measurements are selected. Inequalities are preserved. Activity binary
uses <=4 µM positive, panel <=16 µM positive, and both use >32 µM negative.
Hemolysis keeps the source audit's explicit percent-band rule and risk direction.
Each sequence/output requires unanimity among selected observations: either
conflicting labels or selected ambiguous evidence masks the entire group.

Excluded mass-unit, range, uncertainty and other observations do not veto a
retained group. This is an explicit subset policy, NOT a claim of agreement across
all experiments. Donor/strain/condition details remain in the original source
audit, linked through source lines and hashes. Binary agreement among retained
contexts does not establish activity or safety in all contexts.

The dataset is molar-only, not verified unmodified. Missing terminal chemistry
remains a limitation. No training runs automatically. Share the console summary
to assess class balance, retention, and family coverage first.

The existing family partition is used only for coverage reporting. Its test has
already been inspected. New sequences could bridge families; do not append them
to existing training splits. The common-population table does not provide
independent ground truth. A controlled label-ablation protocol must be declared
before training, with a common evaluation population and explicit handling of
the already-inspected test.
