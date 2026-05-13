from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


DEFAULT_PATTERNS = (
    "embedding.word_embeddings",
    "decoder.layers.0.input_layernorm",
    "decoder.layers.0.self_attention",
    "decoder.layers.0.pre_mlp_layernorm",
    "decoder.layers.0.mlp.router",
    "decoder.layers.0.mlp.shared_experts",
    "decoder.layers.0.mlp",
    "decoder.final_layernorm",
    "output_layer",
)


def _iter_tensors(value: Any, prefix: str = "out"):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            yield from _iter_tensors(item, f"{prefix}[{idx}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors(item, f"{prefix}.{key}")


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    total = detached.numel()
    summary: dict[str, Any] = {
        "shape": tuple(detached.shape),
        "dtype": str(detached.dtype),
        "finite": finite_count,
        "total": total,
        "nan": int(torch.isnan(detached).sum().item()),
        "inf": int(torch.isinf(detached).sum().item()),
    }
    if finite_count:
        vals = detached[finite].float()
        summary.update(
            {
                "min": float(vals.min().item()),
                "max": float(vals.max().item()),
                "mean": float(vals.mean().item()),
            }
        )
    return summary


def _should_capture_tensor(name: str, tensor: torch.Tensor, max_elems: int) -> bool:
    if "output_layer" in name:
        return True
    return tensor.numel() <= max_elems


def before_log_prob(args, model, store_prefix: str) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    patterns = tuple(
        item.strip()
        for item in os.environ.get("MILES_DEBUG_FORWARD_TRACE_PATTERNS", ",".join(DEFAULT_PATTERNS)).split(",")
        if item.strip()
    )
    max_modules = int(os.environ.get("MILES_DEBUG_FORWARD_TRACE_MAX_MODULES", "64"))
    max_elems = int(os.environ.get("MILES_DEBUG_FORWARD_TRACE_MAX_ELEMS", "2000000"))
    trace_path = Path(args.dump_details) / f"forward_trace_rank{rank}.pt"
    log_path = Path(args.dump_details) / f"forward_trace_rank{rank}.log"
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "rank": rank,
        "store_prefix": store_prefix,
        "patterns": patterns,
        "records": [],
        "seen": set(),
    }

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")
            f.flush()
        print(message, flush=True)

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if name in state["seen"] or len(state["seen"]) >= max_modules:
                return
            record: dict[str, Any] = {"name": name, "tensors": {}}
            captured_any = False
            for tensor_name, tensor in _iter_tensors(output):
                if tensor.numel() == 0:
                    continue
                entry: dict[str, Any] = {"summary": _tensor_summary(tensor)}
                if _should_capture_tensor(name, tensor, max_elems):
                    entry["value"] = tensor.detach().cpu()
                record["tensors"][tensor_name] = entry
                captured_any = True
            if not captured_any:
                return
            state["seen"].add(name)
            state["records"].append(record)
            torch.save(state, trace_path)
            log(f"[forward_trace] rank={rank} captured={name} tensors={list(record['tensors'])}")

        return hook

    registered = 0
    for model_idx, model_module in enumerate(model):
        for module_name, module in model_module.named_modules():
            full_name = f"model[{model_idx}].{module_name}" if module_name else f"model[{model_idx}]"
            if not any(pattern in full_name for pattern in patterns):
                continue
            module.register_forward_hook(make_hook(full_name))
            registered += 1

    torch.save(state, trace_path)
    log(f"[forward_trace] rank={rank} registered={registered} store_prefix={store_prefix!r}")
