"""Create a durable byte-level repeatability bundle from two strict runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiment_utils import sha256, write_json

OUTPUTS = {"library": "library.fasta", "top": "top.fasta", "top_scores": "top_scores.csv"}
MANIFEST_KINDS = {"amp_generation_manifest_v1", "amp_generation_manifest_v2"}


def _verified_run(root: Path) -> tuple[dict, dict[str, str]]:
    manifest_path = root / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") not in MANIFEST_KINDS or manifest.get("status") != "complete":
        raise ValueError(f"Invalid strict generation manifest: {manifest_path}")
    output_hashes = {}
    for name, relative in OUTPUTS.items():
        record = manifest.get("outputs", {}).get(name)
        if not isinstance(record, dict):
            raise ValueError(f"Manifest omits {name}: {manifest_path}")
        path = (root / record.get("path", "")).resolve()
        if root.resolve() not in path.parents or not path.is_file() or sha256(path) != record.get("sha256"):
            raise ValueError(f"Output does not match strict manifest: {path}")
        output_hashes[name] = sha256(path)
    if set(manifest["outputs"]) != set(OUTPUTS):
        raise ValueError("Strict manifest contains missing or unexpected outputs")
    return manifest, output_hashes


def _recipe(manifest: dict) -> dict:
    recipe = dict(manifest.get("recipe", {}))
    # Strict generation must use separate output directories; this one field is
    # the only recipe difference ignored during repeatability comparison.
    recipe.pop("out_dir", None)
    return recipe


def run(first: Path, second: Path, out: Path) -> Path:
    first, second, out = first.resolve(), second.resolve(), out.resolve()
    for source in (first, second):
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("Output bundle must be separate from both input runs")
    if first == second or out.exists():
        raise ValueError("Select two distinct source runs and a new output directory")
    manifest1, hashes1 = _verified_run(first)
    manifest2, hashes2 = _verified_run(second)
    if _recipe(manifest1) != _recipe(manifest2):
        raise ValueError("Strict run effective recipes differ beyond the documented output directory")
    equal = hashes1 == hashes2
    out.mkdir(parents=True)
    for name, source in (("run1", first), ("run2", second)):
        target = out / name
        target.mkdir()
        shutil.copyfile(source / "generation_manifest.json", target / "generation_manifest.json")
        for logical, relative in OUTPUTS.items():
            record = json.loads((source / "generation_manifest.json").read_text())["outputs"][logical]
            shutil.copyfile(source / record["path"], target / OUTPUTS[logical])
    comparison = {
        "kind": "strict_generation_repeatability_v1", "outputs_equal": equal,
        "run1_manifest_sha256": sha256(out / "run1/generation_manifest.json"),
        "run2_manifest_sha256": sha256(out / "run2/generation_manifest.json"),
        "run1_output_sha256": hashes1, "run2_output_sha256": hashes2,
        "runtime_equal": {key: manifest1.get(key) for key in ("runtime", "device", "scorers")} ==
                         {key: manifest2.get(key) for key in ("runtime", "device", "scorers")},
        "runtime1": {key: manifest1.get(key) for key in ("runtime", "device", "scorers")},
        "runtime2": {key: manifest2.get(key) for key in ("runtime", "device", "scorers")},
        "runtime_limitation": "Byte equality is reported for these runs; runtime differences do not establish cross-runtime reproducibility.",
        "ignored_recipe_field": "out_dir",
    }
    write_json(out / "comparison.json", comparison)
    names = ["comparison.json"]
    for run_name in ("run1", "run2"):
        names += [f"{run_name}/generation_manifest.json", *[f"{run_name}/{value}" for value in OUTPUTS.values()]]
    write_json(out / "complete.json", {"files": {name: sha256(out / name) for name in names}})
    if not equal:
        raise ValueError(f"Strict runs differ; report preserved at {out}")
    print(f"Verified byte-identical strict runs: {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run1", type=Path, required=True)
    parser.add_argument("--run2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.run1, args.run2, args.out)


if __name__ == "__main__":
    main()
