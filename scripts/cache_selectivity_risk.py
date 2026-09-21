"""Build a resumable, source-bound hemolysis cache for frozen hybrid libraries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from experiment_utils import (
    REPO_ROOT,
    code_identity,
    mark_files,
    prepare_run,
    sha256,
    verify_files,
    write_json,
)
from selection_cache import load_source, separate_output, verify_source

from amp_challenge_2027.selectivity import sequence_digest, validate_protocol, validate_risk


def _protocol(path: Path) -> dict:
    data = json.loads(path.read_text())
    validate_protocol(data)
    return data


def _build_union(cells: list[dict]) -> tuple[list[str], dict[int, dict]]:
    hybrid = sorted((cell for cell in cells if cell["case"] == "hybrid"), key=lambda cell: cell["seed"])
    if [cell["seed"] for cell in hybrid] != [42, 43, 44]:
        raise ValueError("Expected hybrid generation seeds 42, 43, 44")
    union: list[str] = []
    index: dict[str, int] = {}
    maps = {}
    for cell in hybrid:
        mapped = np.empty(len(cell["sequences"]), dtype=np.int64)
        for i, sequence in enumerate(cell["sequences"]):
            if sequence not in index:
                index[sequence] = len(union)
                union.append(sequence)
            mapped[i] = index[sequence]
        maps[cell["seed"]] = {"cell": cell, "indices": mapped}
    return union, maps


def _risk_identity(args, protocol_path: Path, protocol: dict, source_identity: dict,
                   union: list[str], cell_maps: dict[int, dict], artifacts: dict) -> dict:
    return {
        "kind": "selectivity_risk_cache_v1",
        "source_run_sha256": source_identity["run_sha256"],
        "source_root": str(Path(source_identity["source"]).resolve()),
        "protocol_sha256": sha256(protocol_path),
        "reference_sha256": sha256(args.reference),
        "union_sequence_sha256": sequence_digest(union),
        "union_unique_sequences": len(union),
        "cells": {str(seed): {"library_sha256": item["cell"]["record"]["library_sha256"],
                              "index_map_sha256": sequence_digest([str(int(i)) for i in item["indices"]]),
                              "occurrences": len(item["indices"])}
                  for seed, item in sorted(cell_maps.items())},
        "hemolysis_head_files": artifacts["hemolysis"]["files"],
        "hemolysis_config": artifacts["hemolysis"]["config"],
        "temperature": artifacts["hemolysis"]["config"]["temperature"],
        "label_meaning": "p(risky under the frozen candidate hemolysis label definition)",
        "backbone_revision": source_identity["backbones"]["hemolysis"],
        "preprocessing": {"tokenizer": artifacts["hemolysis"]["config"]["esm_model"],
                          "max_length": 52, "truncation": True, "padding": True,
                          "batch_size": args.batch_size},
        "runtime": {"code": code_identity(), "device": args.device,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "platform": sys.platform},
    }


def _load_chunks(out: Path, identity: dict) -> tuple[np.ndarray, np.ndarray]:
    n = identity["union_unique_sequences"]
    values = np.full(n, np.nan, dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    chunk_dir = out / "chunks"
    if not chunk_dir.exists():
        return values, done
    for marker_path in sorted(chunk_dir.glob("*.json")):
        marker = json.loads(marker_path.read_text())
        data_path = chunk_dir / marker.get("array", "")
        start, end = marker.get("start"), marker.get("end")
        if (not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start
                or end > n or Path(marker.get("array", "")).name != marker.get("array")
                or data_path.resolve().parent != chunk_dir.resolve()
                or not data_path.is_file() or sha256(data_path) != marker.get("sha256")):
            raise ValueError(f"Corrupt marked risk chunk: {marker_path}")
        chunk = np.load(data_path, allow_pickle=False)
        validate_risk(chunk, end - start)
        if done[start:end].any():
            raise ValueError(f"Overlapping marked risk chunks: {marker_path}")
        values[start:end] = chunk
        done[start:end] = True
    return values, done


def run(args, *, scorer=None) -> int:
    args.source, args.protocol, args.reference, args.out = map(
        lambda path: path.resolve(), (args.source, args.protocol, args.reference, args.out)
    )
    protocol = _protocol(args.protocol)
    if args.source != (REPO_ROOT / protocol["source"]).resolve():
        raise ValueError("Source path differs from the frozen protocol")
    source_identity, all_cells, _reference = load_source(args.source, args.reference)
    cells = [cell for cell in all_cells if cell["case"] == "hybrid"]
    union, maps = _build_union(cells)
    separate_output(args.out, [args.source, args.protocol, args.reference])
    artifacts = __import__("compare_top100").classifier_inputs()
    source_hemo = source_identity["source_run"]["classifiers"]["hemolysis"]
    if artifacts["hemolysis"]["files"] != source_hemo["files"]:
        raise ValueError("Deployed hemolysis head/config hashes differ from the frozen source")
    if source_identity["backbones"].get("hemolysis") is None:
        raise ValueError("Frozen source lacks an immutable resolved hemolysis backbone revision")
    recipe = _risk_identity(args, args.protocol, protocol, source_identity, union, maps, artifacts)
    print(f"[risk-cache] ordered unique union: {len(union)} sequences; occurrences: {sum(len(v['indices']) for v in maps.values())}")
    print(f"[risk-cache] cells requested: {args.seeds}; device={args.device}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if args.list:
        print("[risk-cache] preflight only; --list validates source/protocol and does not load a model or write files")
        return 0
    prepare_run(args.out, recipe)
    if (args.out / "complete.json").exists():
        complete = json.loads((args.out / "complete.json").read_text())
        if complete.get("kind") != "selectivity_risk_cache_complete_v1":
            raise ValueError("Invalid risk-cache completion marker")
        for name, expected in complete.get("files", {}).items():
            if not (args.out / name).is_file() or sha256(args.out / name) != expected:
                raise ValueError(f"Completed risk-cache root artifact changed: {name}")
        for seed in (42, 43, 44):
            directory = args.out / "cells/hybrid" / f"seed{seed}"
            verify_files(directory, "complete.json")
        cached_values, cached_done = _load_chunks(args.out, recipe)
        if not cached_done.all():
            raise ValueError("Completed risk cache has incomplete chunk coverage")
        for seed, item in maps.items():
            directory = args.out / "cells/hybrid" / f"seed{seed}"
            indices = np.load(directory / "indices.npy", allow_pickle=False)
            risk_values = validate_risk(np.load(directory / "risk.npy", allow_pickle=False), len(item["indices"]))
            if not np.array_equal(indices, item["indices"]) or not np.array_equal(risk_values, cached_values[indices]):
                raise ValueError(f"Completed risk cell arrays differ from validated chunks: seed {seed}")
        verify_source(source_identity)
        print("[risk-cache] matching complete cache verifies; no model load needed")
        return 0
    if scorer is None:
        from amp_challenge_2027.score import HemoScorer
        scorer = HemoScorer.load(device=args.device, revision=source_identity["backbones"]["hemolysis"])
        if scorer is None:
            raise RuntimeError("Required hemolysis scorer failed to load; stopping")
    if not np.isclose(scorer._temperature, recipe["temperature"], rtol=1e-9, atol=0):
        raise ValueError("Loaded scorer temperature differs from frozen source")
    model = getattr(scorer, "_model", None)
    model = getattr(model, "esm", model)
    config = getattr(model, "config", None)
    loaded_revision = getattr(config, "_commit_hash", None)
    if loaded_revision != recipe["backbone_revision"]:
        raise ValueError("Loaded hemolysis backbone revision differs from the frozen source")

    # Verify cached source top risks before scoring a pool or trusting a resume.
    for seed in args.seeds:
        cell = maps[seed]["cell"]
        expected = np.asarray([float(row["hemo_risk"]) for row in cell["top_rows"]], dtype=np.float32)
        observed = validate_risk(scorer.p_risky(cell["top"]), len(cell["top"]))
        if not np.allclose(observed, expected, rtol=1e-5, atol=1e-5):
            raise ValueError(f"Source top-risk parity failed for hybrid seed {seed}")

    values, done = _load_chunks(args.out, recipe)
    max_index = max(int(maps[seed]["indices"].max()) + 1 for seed in args.seeds)
    missing = np.flatnonzero(~done[:max_index])
    budget = args.max_new_sequences
    if budget is not None:
        missing = missing[:budget]
    chunk_dir = args.out / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    inferred = 0
    for offset in range(0, len(missing), args.batch_size):
        indices = missing[offset:offset + args.batch_size]
        if not len(indices):
            continue
        start, end = int(indices[0]), int(indices[-1]) + 1
        # A gap can occur when deduplication makes mapped indices sparse.
        if not np.array_equal(indices, np.arange(start, end)):
            for index in indices:
                risk = validate_risk(scorer.p_risky([union[int(index)]]), 1)
                _write_chunk(chunk_dir, int(index), int(index) + 1, risk)
                values[index], done[index] = risk[0], True
                inferred += 1
            continue
        risk = validate_risk(scorer.p_risky([union[int(i)] for i in indices]), len(indices))
        _write_chunk(chunk_dir, start, end, risk)
        values[start:end], done[start:end] = risk, True
        inferred += len(indices)
        print(f"[risk-cache] scored {inferred}/{len(missing)} new sequences", flush=True)

    covered = {}
    for seed, item in sorted(maps.items()):
        idx = item["indices"]
        covered[str(seed)] = bool(done[idx].all())
        if covered[str(seed)]:
            cell_dir = args.out / "cells" / "hybrid" / f"seed{seed}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            if (cell_dir / "complete.json").exists():
                verify_files(cell_dir, "complete.json")
                prior = json.loads((cell_dir / "identity.json").read_text())
                if prior.get("sequence_sha256") != sequence_digest(item["cell"]["sequences"]):
                    raise ValueError(f"Completed cell cache identity changed for seed {seed}")
                stored_indices = np.load(cell_dir / "indices.npy", allow_pickle=False)
                stored_risk = validate_risk(np.load(cell_dir / "risk.npy", allow_pickle=False), len(idx))
                if (not np.array_equal(stored_indices, idx) or not np.array_equal(stored_risk, values[idx])
                        or prior.get("risk_sha256") != sha256(cell_dir / "risk.npy")
                        or prior.get("indices_sha256") != sha256(cell_dir / "indices.npy")):
                    raise ValueError(f"Completed cell cache differs from chunks for seed {seed}")
            else:
                np.save(cell_dir / "risk.npy", values[idx].astype(np.float32), allow_pickle=False)
                np.save(cell_dir / "indices.npy", idx.astype(np.int64), allow_pickle=False)
                write_json(cell_dir / "identity.json", {"seed": seed, "library_sha256": item["cell"]["record"]["library_sha256"],
                                                         "sequence_sha256": sequence_digest(item["cell"]["sequences"]),
                                                         "risk_sha256": sha256(cell_dir / "risk.npy"),
                                                         "indices_sha256": sha256(cell_dir / "indices.npy")})
                mark_files(cell_dir, "complete.json", ["risk.npy", "indices.npy", "identity.json"])
    full = all(covered.values())
    write_json(args.out / "coverage.json", {"cells": covered, "complete": full,
                                              "cached_sequences": int(done.sum()), "union_sequences": len(union),
                                              "requested_seeds": args.seeds})
    status = "complete" if full else "incomplete"
    write_json(args.out / "status.json", {"status": status, "newly_inferred": inferred,
                                           "cached_sequences": int(done.sum()), "union_sequences": len(union)})
    files = ["run.json", "coverage.json", "status.json"]
    if full:
        marker_files = {name: sha256(args.out / name) for name in files}
        write_json(args.out / "complete.json", {"kind": "selectivity_risk_cache_complete_v1", "files": marker_files,
                                                 "cells": {str(seed): {"complete_sha256": sha256(args.out / "cells/hybrid" / f"seed{seed}" / "complete.json"),
                                                                       "risk_sha256": sha256(args.out / "cells/hybrid" / f"seed{seed}" / "risk.npy")}
                                                            for seed in (42, 43, 44)}})
    verify_source(source_identity)
    print(f"[risk-cache] status={status}; newly inferred={inferred}; cached={int(done.sum())}/{len(union)}")
    return 0 if all(covered[str(seed)] for seed in args.seeds) else 3


def _write_chunk(directory: Path, start: int, end: int, values: np.ndarray) -> None:
    name = f"{start:09d}-{end:09d}.npy"
    path = directory / name
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values.astype(np.float32), allow_pickle=False)
    temporary.replace(path)
    write_json(directory / f"{start:09d}-{end:09d}.json",
               {"start": start, "end": end, "array": name, "sha256": sha256(path)})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("data/antibacterial.fasta"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, choices=[42, 43, 44], default=[42])
    parser.add_argument("--max-new-sequences", type=int)
    parser.add_argument("--list", action="store_true", help="Validate source/protocol only; no model or output resumability check")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or (args.max_new_sequences is not None and args.max_new_sequences <= 0):
        parser.error("batch size and sequence budget must be positive")
    args.seeds = list(dict.fromkeys(args.seeds))
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
