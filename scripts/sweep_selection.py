"""Selection-strategy sweep: composite-weight grid x pool mixes -> Phase-1 metrics.

Evaluates many selection configurations against the official protocol WITHOUT
re-running model inference: pools are loaded and component scores are computed
exactly once, then each grid cell just re-weights the cached components,
re-selects, writes a library/top pair, and (optionally) scores it with seqme
using one shared ESM-2 embedder.

Grid semantics:
    --grid  "a,c,p" triples separated by ';'   (composite weights)
    --mixes comma-lists of pool indices ';'   ("0" = seed44 only, "0,1" = blend)

Outputs per cell ``<out>/<tag>/``{library,top}.fasta plus one aggregated CSV
(``results.csv``) and a Pareto-sorted console table.

Run (GPU box):
    uv run --extra ml --extra seqme python scripts/sweep_selection.py \
        --pools generate/pool-seed44.fasta generate/pool-v2.fasta generate/pool-seed51.fasta \
        --device cuda --out sweep_results/selection

Run (local smoke test, no torch/seqme needed):
    uv run python scripts/sweep_selection.py --pools some.fasta --smoke --out /tmp/sweep-smoke
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from amp_challenge_2027.config import ANTIBACTERIAL_FASTA, DEFAULT_SEED, LIBRARY_SIZE, TOP_K
from amp_challenge_2027.data import read_reference_set, write_fasta
from amp_challenge_2027.pipeline import (
    DEFAULT_WEIGHTS,
    build_composite_scorer,
    clean_candidates,
    load_pool_fastas,
)
from amp_challenge_2027.select import select_library_and_top


def parse_triples(raw: str) -> list[tuple[float, ...]]:
    """Grid cells: 3-tuples (activity,conformity,precision) legacy form, or
    5-tuples (...,breadth,mdr) for panel-weighted cells."""
    cells = []
    for chunk in raw.split(";"):
        vals = tuple(float(x) for x in chunk.split(","))
        if len(vals) not in (3, 5):
            raise ValueError(
                f"grid cell {chunk!r} must have 3 weights (a,c,p) or 5 (a,c,p,breadth,mdr)"
            )
        cells.append(vals)
    return cells


def parse_mixes(raw: str, n_sources: int) -> list[list[int]]:
    mixes = []
    for chunk in raw.split(";"):
        idx = [int(x) for x in chunk.split(",") if x != ""]
        for i in idx:
            if not 0 <= i < n_sources:
                raise ValueError(f"mix references pool {i}, but only {n_sources} pools given")
        mixes.append(sorted(set(idx)))
    return mixes


def _z(vals: np.ndarray) -> np.ndarray:
    sd = vals.std()
    return (vals - vals.mean()) / (sd if sd > 1e-8 else 1.0)


def _scalarize(row: dict, name: str, val) -> None:
    """Store a seqme metric cell as float(s); multi-value cells get ``name.N``.

    Several seqme metrics return (value, deviation)-style pairs; the primary
    value keeps the metric name so downstream tables/sorting are stable.
    """
    try:
        row[name] = float(val)
        return
    except (TypeError, ValueError):
        pass
    arr = np.asarray(val, dtype=float).ravel()
    row[name] = float(arr[0])
    for i, v in enumerate(arr[1:], start=1):
        row[f"{name}.{i}"] = float(v)


def _collect_metrics(row: dict, df, *, dataset_name: str | None = None) -> list[str]:
    """Flatten a seqme ``evaluate`` frame into ``row``; returns metric names.

    Handles BOTH observed layouts:
      wide  — index = dataset rows (e.g. 'library'), columns = MultiIndex of
              (metric, statistic)          [current seqme behavior]
      tall  — index = metric names, columns = (group, statistic) MultiIndex

    The primary statistic (value/mean) keeps the bare metric name; other
    statistics become ``<metric>.<stat>`` keys. Metric names always come from
    the FRAME — never assumed (they drifted once already).
    """

    def stat_name(c) -> str:
        return str(c[-1]).strip() if isinstance(c, tuple) else str(c)

    cols = list(df.columns)
    names: list[str] = []

    if (
        cols
        and isinstance(cols[0], tuple)
        and df.index.tolist()
        and not isinstance(df.index[0], tuple)
    ):
        # WIDE: one row per dataset; metric lives in the column's first level.
        by_metric: dict[str, list] = {}
        for c in cols:
            by_metric.setdefault(str(c[0]), []).append(c)
        key = dataset_name or str(df.index[0])
        for metric, cs in by_metric.items():
            primary = next(
                (
                    c
                    for c in cs
                    if "value" in stat_name(c).lower() or "mean" in stat_name(c).lower()
                ),
                cs[0],
            )
            _scalarize(row, metric, df.loc[key, primary])
            for c in cs:
                if c != primary:
                    _scalarize(row, f"{metric}.{stat_name(c).replace(' ', '_')}", df.loc[key, c])
            names.append(metric)
        return names

    # TALL: metric per index row.
    primary = 0
    for i, c in enumerate(cols):
        low = stat_name(c).lower()
        if "value" in low or "mean" in low:
            primary = i
            break

    for metric_name in df.index:
        vals = df.loc[metric_name]
        base = str(metric_name)
        _scalarize(row, base, vals.iloc[primary])
        for i, c in enumerate(cols):
            if i != primary:
                suffix = stat_name(c).replace(" ", "_")
                _scalarize(row, f"{base}.{suffix}", vals.iloc[i])
        names.append(base)
    return names


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Composite-selection sweep.")
    parser.add_argument("--pools", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--grid",
        type=str,
        default="1,0.5,0.5;1,0.5,0;1,0,0.5;0,0.5,0.5;1,0.25,0.75;1,0.75,0.25;1,1,1",
        help="';'-separated weight cells: activity,conformity,precision "
        "(legacy 3-tuple) or activity,conformity,precision,breadth,mdr "
        "(5-tuple, panel-weighted; requires classifier_panel.pt)",
    )
    parser.add_argument(
        "--mixes",
        type=str,
        default=None,
        help="';'-separated pool-index mixes (default: each pool alone, then all together)",
    )
    parser.add_argument("--reference", type=Path, default=ANTIBACTERIAL_FASTA)
    parser.add_argument("--library-size", type=int, default=LIBRARY_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conformity-sample", type=int, default=12000)
    parser.add_argument("--precision-esm", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--novelty-candidates", type=int, default=2000)
    parser.add_argument("--pool-cap-per-source", type=int, default=0)
    parser.add_argument("--eval", dest="eval_metrics", action="store_true", default=True)
    parser.add_argument("--no-eval", dest="eval_metrics", action="store_false")
    parser.add_argument(
        "--esm-model",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="embedder for the official metrics",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny sizes + cheap metrics; runs in the minimal env without torch/seqme",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.smoke:
        args.library_size = min(args.library_size, 400)
        args.top_k = min(args.top_k, 50)
        args.conformity_sample = min(args.conformity_sample, 800)
        args.novelty_candidates = min(args.novelty_candidates, 200)

    reference_set: set[str] = set()
    if args.reference.exists():
        reference_set = read_reference_set(args.reference)
        print(f"[sweep] loaded {len(reference_set)} reference sequences")
    else:
        print(f"[sweep] WARNING: {args.reference} missing; overlap/novelty checks skipped")

    # --- Load + clean each source separately, then a global clean union -----
    per_source_clean: list[list[str]] = []
    for i, path in enumerate(args.pools):
        src = load_pool_fastas(
            [path], cap_per_source=args.pool_cap_per_source, seed=args.seed + i * 1000
        )
        per_source_clean.append(clean_candidates(src, reference_set))
        print(f"[sweep] pool[{i}] {path.name}: {len(per_source_clean[-1])} clean")

    union = [s for src in per_source_clean for s in src]
    clean_union = clean_candidates(union, reference_set)
    print(f"[sweep] union: {len(clean_union)} clean unique candidates")

    # --- Component scores computed exactly once over the union --------------
    # breadth/mdr are requested with weight 1 purely so the components get
    # registered and cached; per-cell weights come from the grid. When the
    # panel artifact is absent they silently drop and 5-tuple cells just
    # renormalize over the remaining components.
    scorer = build_composite_scorer(
        sorted(reference_set),
        w_activity=DEFAULT_WEIGHTS["activity"],
        w_conformity=DEFAULT_WEIGHTS["conformity"],
        w_precision=DEFAULT_WEIGHTS["precision"],
        w_breadth=1.0,
        w_mdr=1.0,
        device=args.device,
        conformity_sample=args.conformity_sample,
        precision_esm_model=args.precision_esm,
        seed=args.seed,
    )
    part_values: dict[str, np.ndarray]
    if scorer is not None:
        print(f"[sweep] computing components {scorer.names} once over the union...")
        t0 = time.time()
        _, part_values = scorer.score(clean_union)
        print(f"[sweep] components done in {time.time() - t0:.1f}s")
    else:
        part_values = {}
        print("[sweep] no scoring components available; cells differ only by mix")

    seq_pos = {s: i for i, s in enumerate(clean_union)}
    grid = parse_triples(args.grid)
    mixes = (
        parse_mixes(args.mixes, len(args.pools))
        if args.mixes
        else [[i] for i in range(len(args.pools))] + [list(range(len(args.pools)))]
    )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol_metric_names: list[str] = []  # discovered from the first eval frame
    rows: list[dict] = []

    embedder = None
    metric_objs: list = []
    if args.eval_metrics and not args.smoke:
        try:
            import seqme as sm

            from amp_challenge_2027.metrics_official import build_metric_list

            print(f"[sweep] loading eval embedder {args.esm_model}...")
            embedder = sm.models.ESM2(model_name=args.esm_model, device=args.device)
            metric_pairs = build_metric_list(sorted(reference_set), embedder)
            metric_objs = [m for _, m in metric_pairs]
        except ImportError as e:
            print(f"[sweep] seqme unavailable ({e}); writing libraries without metrics")

    for mi, mix in enumerate(mixes):
        mix_seqs_all = [s for i in mix for s in per_source_clean[i]]
        # Cross-source dedup, keep first occurrence in mix order.
        seen: set[str] = set()
        mix_seqs: list[str] = []
        for s in mix_seqs_all:
            if s not in seen:
                seen.add(s)
                mix_seqs.append(s)

        pos = np.array([seq_pos[s] for s in mix_seqs], dtype=np.int64)
        mix_parts = {name: vals[pos] for name, vals in part_values.items()}

        for cell in grid:
            wa, wc, wp = cell[0], cell[1], cell[2]
            wb = cell[3] if len(cell) == 5 else 0.0
            wm = cell[4] if len(cell) == 5 else 0.0
            weights = {
                "activity": wa,
                "conformity": wc,
                "precision": wp,
                "breadth": wb,
                "mdr": wm,
            }
            combined = np.zeros(len(mix_seqs), dtype=np.float64)
            total_w = 0.0
            for name, w in weights.items():
                if w and name in mix_parts:
                    combined += w * _z(mix_parts[name])
                    total_w += w
            scores = (combined / total_w).astype(np.float32) if total_w > 0 else None

            result = select_library_and_top(
                mix_seqs,
                reference_set=reference_set,
                scores=scores,
                top_k=args.top_k,
                library_size=args.library_size,
                seed=args.seed,
                max_novelty_candidates=args.novelty_candidates,
            )
            suffix = f"-{wb:g}-{wm:g}" if len(cell) == 5 else ""
            tag = f"m{mi}_w{wa:g}-{wc:g}-{wp:g}{suffix}"
            cell_dir = out_dir / tag
            write_fasta(result.library, cell_dir / "library.fasta")
            write_fasta(result.top, cell_dir / "top.fasta")

            row = {
                "cell": tag,
                "mix": "+".join(str(i) for i in mix),
                "w_activity": wa,
                "w_conformity": wc,
                "w_precision": wp,
                "w_breadth": wb,
                "w_mdr": wm,
                "library_size": len(result.library),
            }
            if metric_objs:
                import seqme as sm

                try:
                    df = sm.evaluate({"library": result.library}, metric_objs)
                except Exception as e:
                    print(
                        f"[sweep] WARNING: evaluation failed for {tag} ({e}); "
                        "cell written without metrics"
                    )
                    df = None
                if df is not None:
                    discovered = _collect_metrics(row, df, dataset_name="library")
                    if not protocol_metric_names:
                        protocol_metric_names = discovered
                        print(
                            f"[sweep] protocol metrics discovered: "
                            f"{len(protocol_metric_names)} — {protocol_metric_names}"
                        )
            rows.append(row)
            got = {k: round(v, 4) for k, v in row.items() if k in protocol_metric_names}
            print(f"[sweep] {tag}: lib={len(result.library)} {got}")

    # --- Aggregate -----------------------------------------------------------
    csv_path = out_dir / "results.csv"
    fieldnames = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sweep] wrote {csv_path} ({len(rows)} cells)")

    if protocol_metric_names:
        sort_key = "FBD" if "FBD" in protocol_metric_names else protocol_metric_names[0]
        printable = [r for r in rows if sort_key in r]
        printable.sort(key=lambda r: r[sort_key])
        cols = [
            "cell",
            "mix",
            *[
                m
                for m in ("FBD", "MMD", "Precision", "Recall", "ConformityScore", "AuthPct")
                if m in protocol_metric_names
            ],
        ]
        print(f"\n=== Pareto table (sorted by {sort_key}) ===")
        header = "  ".join(f"{c[:12]:>12}" for c in cols)
        print(header)
        for r in printable:
            print(
                "  ".join(
                    f"{r.get(c, float('nan')):>12.4f}"
                    if isinstance(r.get(c), float)
                    else f"{str(r.get(c, ''))[:12]:>12}"
                    for c in cols
                )
            )


if __name__ == "__main__":
    main()
