"""Profiling harness for the transformer training pipeline.

Profiles one forward+backward pass and (optionally) a few real training steps,
reporting per-op GPU time, memory, and FLOPs. Output goes to the console, a
JSON summary, and a W&B system metrics panel if W&B is enabled.

This is a diagnostic tool — run it once before a full training run to:
  - Catch OOMs at the intended batch size / precision before wasting hours
  - Identify whether attention, FFN, or MoE routing is the bottleneck
  - Verify that bf16 actually speeds things up vs fp32 on your hardware
  - Measure the real tokens/sec to estimate epoch wall-clock time

Run:
    uv run --extra ml python scripts/profile_train.py \\
        --batch-size 128 --precision bf16 --seq-length 52
    uv run --extra ml python scripts/profile_train.py --moe --num-experts 8
    uv run --extra ml python scripts/profile_train.py --residual block_attnres
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from amp_challenge_2027 import tokenizer as tok
from amp_challenge_2027.config import MAX_LENGTH


def _build_inputs(batch_size: int, seq_length: int, device: str):
    """Random input ids + labels for profiling (residue tokens only)."""
    import torch

    # Random ids in the AA range (4..23), padded with BOS/EOS at ends.
    ids = torch.randint(4, 24, (batch_size, seq_length), device=device)
    ids[:, 0] = tok.BOS_ID
    ids[:, -1] = tok.EOS_ID
    labels = ids.clone()
    return ids, labels


def profile_forward_backward(
    *,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    residual: str,
    ffn: str,
    moe_num_experts: int,
    moe_active: int,
    attnres_blocks: int,
    batch_size: int,
    seq_length: int,
    precision: str,
    device: str,
    warmup: int,
    iters: int,
) -> dict:
    """Profile forward+backward passes. Returns a metrics dict."""
    import torch
    from torch.optim import AdamW

    from amp_challenge_2027.generator import build_model
    from amp_challenge_2027.model import DecoderConfig

    torch.manual_seed(0)
    cfg = DecoderConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        residual=residual,
        ffn=ffn,
        moe_num_experts=moe_num_experts,
        moe_num_active=moe_active,
        attnres_num_blocks=attnres_blocks,
        max_position_embeddings=seq_length,
    )
    model, _ = build_model(cfg)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    autocast_ctx = _make_autocast(precision, device)
    scaler = torch.amp.GradScaler(device) if precision == "fp16" else None

    input_ids, labels = _build_inputs(batch_size, seq_length, device)

    # --- Warmup (not timed) ------------------------------------------------
    print(f"[profile] warming up ({warmup} iters)...")
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            out = model(input_ids)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = loss + out.aux_loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # --- Timed run ---------------------------------------------------------
    print(f"[profile] timed run ({iters} iters)...")
    timings = []
    peak_mem_before = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with autocast_ctx():
            out = model(input_ids)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = lm_loss + out.aux_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    peak_mem = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
    peak_mem_mb = peak_mem / 1e6
    mem_delta_mb = (peak_mem - peak_mem_before) / 1e6

    timings_arr = timings[1:] if len(timings) > 1 else timings  # drop first as extra warmup
    avg_ms = 1000 * (sum(timings_arr) / len(timings_arr))
    tokens_per_iter = batch_size * seq_length
    tokens_per_sec = tokens_per_iter / (avg_ms / 1000)

    return {
        "config": {
            "num_layers": num_layers, "hidden_size": hidden_size, "num_heads": num_heads,
            "residual": residual, "ffn": ffn,
            "moe_num_experts": moe_num_experts if ffn == "moe" else 0,
            "moe_active": moe_active if ffn == "moe" else 0,
            "attnres_blocks": attnres_blocks if "attnres" in residual else 0,
        },
        "n_params": n_params,
        "n_trainable": n_trainable,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "precision": precision,
        "avg_step_ms": round(avg_ms, 3),
        "tokens_per_sec": round(tokens_per_sec, 0),
        "peak_memory_mb": round(peak_mem_mb, 1),
        "memory_delta_mb": round(mem_delta_mb, 1),
        "lm_loss": round(lm_loss.item(), 4),
        "aux_loss": round(out.aux_loss.item(), 4),
    }


def profile_with_torch_profiler(
    *,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    residual: str,
    ffn: str,
    moe_num_experts: int,
    moe_active: int,
    attnres_blocks: int,
    batch_size: int,
    seq_length: int,
    precision: str,
    device: str,
    out_dir: Path,
) -> Path:
    """Run torch.profiler for one step, producing a Chrome trace.

    The trace (``chrome_trace.json``) can be loaded at chrome://tracing or
    tensorboard's profiler tab to see per-op GPU time and kernel launches.
    Useful for identifying whether attention, FFN, or MoE routing dominates.
    """
    import torch
    from torch.optim import AdamW

    from amp_challenge_2027.generator import build_model
    from amp_challenge_2027.model import DecoderConfig

    torch.manual_seed(0)
    cfg = DecoderConfig(
        num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads,
        residual=residual, ffn=ffn, moe_num_experts=moe_num_experts,
        moe_num_active=moe_active, attnres_num_blocks=attnres_blocks,
        max_position_embeddings=seq_length,
    )
    model, _ = build_model(cfg)
    model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    autocast_ctx = _make_autocast(precision, device)
    input_ids, labels = _build_inputs(batch_size, seq_length, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "chrome_trace.json"

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # 3 iterations: 2 warmup + 1 profiled.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                out = model(input_ids)
                shift_logits = out.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                ) + out.aux_loss
            loss.backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            out = model(input_ids)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            ) + out.aux_loss
        loss.backward()
        optimizer.step()

    prof.export_chrome_trace(str(trace_path))
    return trace_path


def _make_autocast(precision: str, device: str):
    from contextlib import nullcontext

    import torch

    if precision == "fp32" or not device.startswith("cuda"):
        return nullcontext
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return lambda: torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the transformer training step.")
    # Architecture (defaults match the SFT defaults)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--residual", type=str, default="standard",
                        choices=["standard", "attnres", "block_attnres"])
    parser.add_argument("--ffn", type=str, default="dense", choices=["dense", "moe"])
    parser.add_argument("--moe", action="store_true", help="shorthand for --ffn moe")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--moe-active", type=int, default=2)
    parser.add_argument("--attnres-blocks", type=int, default=4)
    # Training shape
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-length", type=int, default=MAX_LENGTH + 2)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    # Profiling controls
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--trace", action="store_true",
                        help="also produce a Chrome trace via torch.profiler")
    parser.add_argument("--out-dir", type=Path, default=Path("profiling"))
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="log the summary to W&B (requires wandb installed)")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    ffn = "moe" if args.moe else args.ffn

    # --- Timed profiling ---------------------------------------------------
    metrics = profile_forward_backward(
        num_layers=args.num_layers, hidden_size=args.hidden_size, num_heads=args.num_heads,
        residual=args.residual, ffn=ffn, moe_num_experts=args.num_experts,
        moe_active=args.moe_active, attnres_blocks=args.attnres_blocks,
        batch_size=args.batch_size, seq_length=args.seq_length,
        precision=args.precision, device=args.device,
        warmup=args.warmup, iters=args.iters,
    )

    print("\n" + "=" * 60)
    print("PROFILING SUMMARY")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    print("=" * 60)
    cfg_tag = metrics["config"]
    print(f"\nModel: {cfg_tag['num_layers']}L × {cfg_tag['hidden_size']}H, "
          f"{metrics['n_params']/1e6:.1f}M params ({cfg_tag['residual']}+{cfg_tag['ffn']})")
    print(f"Step time: {metrics['avg_step_ms']:.2f} ms | "
          f"Throughput: {metrics['tokens_per_sec']:,.0f} tokens/sec")
    print(f"Peak GPU memory: {metrics['peak_memory_mb']:.0f} MB "
          f"(+{metrics['memory_delta_mb']:.0f} MB during profiling)")

    # --- Save JSON summary -------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved summary → {summary_path}")

    # --- Optional Chrome trace ---------------------------------------------
    if args.trace:
        print("\n[profile] generating Chrome trace...")
        trace_path = profile_with_torch_profiler(
            num_layers=args.num_layers, hidden_size=args.hidden_size, num_heads=args.num_heads,
            residual=args.residual, ffn=ffn, moe_num_experts=args.num_experts,
            moe_active=args.moe_active, attnres_blocks=args.attnres_blocks,
            batch_size=args.batch_size, seq_length=args.seq_length,
            precision=args.precision, device=args.device, out_dir=args.out_dir,
        )
        print(f"Saved trace → {trace_path} (load at chrome://tracing)")

    # --- Optional W&B logging ----------------------------------------------
    if args.wandb_project:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"profile-{cfg_tag['residual']}-{cfg_tag['ffn']}",
            config=metrics["config"] | {"batch_size": args.batch_size, "precision": args.precision},
        )
        wandb.log({
            "profiling/step_ms": metrics["avg_step_ms"],
            "profiling/tokens_per_sec": metrics["tokens_per_sec"],
            "profiling/peak_memory_mb": metrics["peak_memory_mb"],
            "profiling/n_params": metrics["n_params"],
        })
        if args.trace:
            wandb.save(str(args.out_dir / "chrome_trace.json"))
        wandb.finish()
        print("Logged to W&B.")


if __name__ == "__main__":
    main()
