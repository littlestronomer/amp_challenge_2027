"""Verify deployed metadata, optionally replay fixed reconstruction probes."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from compare_top100 import checked_stage
from experiment_utils import sha256, write_json

from amp_challenge_2027.config import PROJECT_ROOT, REWARD_DIR, REWARD_HEMO_DIR
from amp_challenge_2027.inference_metadata import load_config
from amp_challenge_2027.score import ActivityScorer, HemoScorer, PanelScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a new output directory; existing reports are preserved")
    protocol = json.loads((PROJECT_ROOT / "experiments/reward_reconstruction_v1.json").read_text())
    report = {"kind": "inference_parity_v1", "generalization_verified": False, "tasks": {}}
    for name, root, stem, cls in [
        ("activity", REWARD_DIR, "classifier", ActivityScorer),
        ("panel", REWARD_DIR, "classifier_panel", PanelScorer),
        ("hemolysis", REWARD_HEMO_DIR, "classifier", HemoScorer),
    ]:
        digest = sha256(root / f"{stem}.pt")
        if digest != protocol["tasks"][name]["head_sha256"]:
            raise ValueError(f"{name}: deployed head differs from audited artifact")
        config = load_config(root, stem)
        item = {"head_sha256": digest, "config": config, "probe_verified": False}
        if args.reconstruction:
            directory = args.reconstruction / name
            marker = checked_stage(directory, "complete.json", {"backbone.json", "validation_predictions.csv"})
            backbone = json.loads((directory / "backbone.json").read_text())
            if backbone.get("model") != config["esm_model"]:
                raise ValueError("Reconstruction backbone model differs from deployed model")
            revision = backbone["resolved_revision"]
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Backbone revision must be an immutable commit")
            with (directory / "validation_predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            orders = sorted({int(row["validation_order"]) for row in rows})[:8]
            outputs = config.get("genera", [rows[0]["output"]])
            sequences, expected = [], []
            for order in orders:
                selected = [row for row in rows if int(row["validation_order"]) == order]
                mapping = {row["output"]: row for row in selected}
                if len(mapping) != len(outputs) or len({row["sequence"] for row in selected}) != 1:
                    raise ValueError("Malformed probe records")
                sequences.append(selected[0]["sequence"])
                expected.append([float(mapping[o]["stored_temperature_probability"]) for o in outputs])
            if len(sequences) != 8:
                raise ValueError("Eight probe records required")
            scorer = cls.load(device=args.device, revision=revision)
            if scorer is None:
                raise RuntimeError(f"Required scorer unavailable: {name}")
            actual = (scorer._probs(sequences) if name == "panel" else
                      scorer.p_risky(sequences) if name == "hemolysis" else scorer.score(sequences))
            actual = np.asarray(actual).reshape(len(sequences), -1)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            item.update(probe_verified=True, revision=revision, source_marker=marker,
                        max_absolute_error=float(np.max(np.abs(actual - expected))))
        report["tasks"][name] = item
        print(f"[parity] {name}: metadata verified; probe={item['probe_verified']}")
    write_json(args.out / "report.json", report)


if __name__ == "__main__":
    main()
