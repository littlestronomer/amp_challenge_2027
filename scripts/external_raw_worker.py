"""Run ONLY in an isolated, pinned starter-kit environment (Python 3.8 compatible)."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path


def batch_seed(seed, batch_index):
    """Injective uint32 schedule: distinct runs never share batch seeds."""
    if not 0 <= seed < 65536 or not 0 <= batch_index < 65536:
        raise ValueError("Seed and batch index must fit unsigned 16 bits")
    return (seed << 16) | batch_index


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def sample_diffusion(root, draws, seed, batch_size, device):
    import ampdiffusion_starter_kit.generate as implementation
    import numpy as np
    import torch
    from ampdiffusion_starter_kit.generate import _decode, load_model, set_seed

    if root.resolve() not in Path(implementation.__file__).resolve().parents:
        raise ValueError("AMP-Diffusion import did not resolve to the pinned checkout")

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in AMP-Diffusion environment; no silent CPU fallback")
    set_seed(seed)
    model, esm2, _, indices = load_model(root / "checkpoint/model.pt", torch.device(device))
    model.requires_grad_(False)
    esm2.requires_grad_(False)
    length_rng = np.random.default_rng(seed)
    sequences = []
    with torch.no_grad():
        for start in range(0, draws, batch_size):
            set_seed(batch_seed(seed, start // batch_size))
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            length = int(length_rng.integers(10, 41))
            batch = model.sample(batch_size=min(batch_size, draws-start), design_len=length+2)
            sequences.extend(_decode(esm2, batch, indices, length))
            print(f"[external] diffusion {len(sequences)}/{draws}", flush=True)
    # fair-esm downloads these outside the repository; include their actual hashes.
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    assets = {str(p): digest(p) for p in sorted(cache.glob("esm2_t6_8M_UR50D*.pt"))}
    if not assets:
        raise ValueError("Cannot locate/hash downloaded ESM2 decoder weights")
    return sequences, {"decoder_assets": assets, "device": device, "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
                       "strict_determinism": True}


def sample_hydramp(root, draws, seed, batch_size):
    # Legacy TF2.2 GPU libraries need not match the host CUDA. Small VAE runs on CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import amp.inference.inference as implementation
    from amp.inference.inference import HydrAMPGenerator

    provenance = json.loads(importlib.metadata.distribution("hydramp").read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != "6590d2f4c2963f25d30669052a4c4a857e0e7279":
        raise ValueError("Installed HydrAMP dependency is not the pinned archival commit")

    generator = HydrAMPGenerator(model_path=str(root / "checkpoint/model"),
                                decomposer_path=str(root / "checkpoint/pca_decomposer.joblib"), softmax=True)
    sequences = []
    for start in range(0, draws, batch_size):
        count = min(batch_size, draws-start)
        batch = generator.unconstrained_generation(mode="amp", n_target=count,
                    seed=batch_seed(seed, start // batch_size), filter_out=False, properties=False, n_attempts=1)
        # properties=False preserves draw order, unlike the score-sorted properties=True path.
        if len(batch) != count or not all(isinstance(s, str) for s in batch):
            raise ValueError("HydrAMP violated the exact raw-draw contract")
        sequences.extend(batch)
        print(f"[external] hydramp raw {len(sequences)}/{draws}", flush=True)
    return sequences, {"device": "cpu", "strict_determinism": False,
                       "inference_sha256": digest(implementation.__file__),
                       "warning": "Legacy TF seeded execution; verify repeatability empirically"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("ampdiffusion", "hydramp_raw"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Worker output already exists")
    started = time.monotonic()
    if args.method == "ampdiffusion":
        sequences, runtime = sample_diffusion(args.root, args.draws, args.seed, args.batch_size, args.device)
    else:
        sequences, runtime = sample_hydramp(args.root, args.draws, args.seed, args.batch_size)
    if len(sequences) != args.draws:
        raise ValueError("Wrong raw-draw count")
    packages = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    args.out.write_text(json.dumps({"sequences": sequences, "runtime": runtime, "packages": packages,
                                  "seed_schedule": "uint16_run_uint16_batch_v1", "run_seed": args.seed,
                                  "batch_seeds": [batch_seed(args.seed, i) for i in range((args.draws + args.batch_size - 1) // args.batch_size)],
                                  "python": platform.python_version(), "elapsed_seconds": time.monotonic()-started}))


if __name__ == "__main__":
    main()
