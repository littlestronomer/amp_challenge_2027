"""Re-save the legacy full-body classifier.pt as a head-only checkpoint.

The Aug-2026 binary activity classifier was saved as a FULL state dict
(~135 MB with the ESM-2 t12 backbone) — over GitHub's 100 MB file limit, so
it cannot ship in the repo. ``score.ActivityScorer.load`` already supports
head-only checkpoints (backbone rebuilt from the hub), so this script just
converts the artifact:

  1. loads the full checkpoint into the current ActivityClassifier,
  2. scores a few probe sequences (before-state),
  3. backs up the original to ``classifier.pt.full``,
  4. writes the non-``esm.*`` keys as the new ``classifier.pt``,
  5. reloads via ``ActivityScorer.load`` (the production path) and scores the
     same probes — MAX |Δp| must be < 1e-5 or the script REFUSES to keep the
     conversion (backup restored).

Self-verifying by construction: run it and read the printed verdict.

Run (box):
    uv run python scripts/resave_classifier_head.py \
        [--reward-dir checkpoint/reward] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROBES = [
    "KLLKLLKKLLKL",  # cationic amphipathic archetype
    "GIGKFLHSAKKFGKAFVGEIMNS",  # magainin-like
    "AAADDAAADDAAADD",  # anionic control
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert legacy full classifier.pt to head-only.")
    parser.add_argument("--reward-dir", type=Path, default=Path("checkpoint/reward"))
    parser.add_argument("--dry-run", action="store_true", help="verify only; write nothing")
    args = parser.parse_args(argv)

    ckpt = args.reward_dir / "classifier.pt"
    config_path = args.reward_dir / "config.json"
    if not ckpt.exists() or not config_path.exists():
        print(f"[resave] missing {ckpt} or {config_path}", file=sys.stderr)
        sys.exit(1)

    import torch
    from transformers import AutoModel, AutoTokenizer

    from amp_challenge_2027.score import ActivityScorer, _build_activity_module

    config = json.loads(config_path.read_text())
    sd = torch.load(ckpt, map_location="cpu")
    if not any(k.startswith("esm.") for k in sd):
        print("[resave] checkpoint is already head-only; nothing to do")
        return

    # Rebuild + strict-load the full model to validate the state dict.
    Classifier, _, _ = _build_activity_module(num_outputs=1)
    tokenizer = AutoTokenizer.from_pretrained(config["esm_model"])
    esm = AutoModel.from_pretrained(config["esm_model"])
    model = Classifier(esm.config.hidden_size, 1)
    model.esm = esm
    model.load_state_dict(sd)  # strict: full checkpoint must match exactly
    model.eval()

    def probs(m) -> list[float]:
        import torch as _t

        with _t.no_grad():
            enc = tokenizer(
                PROBES, return_tensors="pt", padding=True, truncation=True, max_length=52
            )
            logits = m(enc["input_ids"], enc["attention_mask"])
            return _t.sigmoid(logits).tolist()

    before = probs(model)
    print(f"[resave] probe probs (full model):   {[round(p, 6) for p in before]}")

    head_sd = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith("esm.")
    }

    if args.dry_run:
        print(
            f"[resave] dry-run: would write {len(head_sd)} head tensors "
            f"(~{sum(v.numel() for v in head_sd.values()) / 1e6:.1f}M params)"
        )
        return

    backup = args.reward_dir / "classifier.pt.full"
    if not backup.exists():
        shutil.copy2(ckpt, backup)
        print(f"[resave] original backed up → {backup}")

    torch.save(head_sd, ckpt)
    new_config = dict(config)
    new_config["checkpoint_format"] = "head-only"
    new_config.setdefault("type", "binary_activity_classifier")
    config_path.write_text(json.dumps(new_config, indent=2))

    # Production-path verification: reload through ActivityScorer.load.
    scorer = ActivityScorer.load(device="cpu")
    if scorer is None:
        shutil.copy2(backup, ckpt)
        print(
            "[resave] FAILED verification (loader returned None); backup restored", file=sys.stderr
        )
        sys.exit(1)
    after = [float(p) for p in scorer.score(PROBES)]
    print(f"[resave] probe probs (head-only):    {[round(p, 6) for p in after]}")
    delta = max(abs(a - b) for a, b in zip(before, after))
    if delta >= 1e-5:
        shutil.copy2(backup, ckpt)
        print(
            f"[resave] FAILED verification (max |Δp| = {delta:.2e}); backup restored",
            file=sys.stderr,
        )
        sys.exit(1)
    size_old = backup.stat().st_size / 1e6
    size_new = ckpt.stat().st_size / 1e6
    print(
        f"[resave] OK: {size_old:.0f} MB → {size_new:.1f} MB "
        f"(max |Δp| = {delta:.2e}); kept backup at {backup.name}"
    )


if __name__ == "__main__":
    main()
