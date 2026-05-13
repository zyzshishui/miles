from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _iter_tensors(value: Any, prefix: str = "out"):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            yield from _iter_tensors(item, f"{prefix}[{idx}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors(item, f"{prefix}.{key}")


def _tensor_stats(tensor: torch.Tensor) -> str:
    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().item())
    total = tensor.numel()
    if finite_count:
        vals = tensor.detach()[finite]
        min_val = float(vals.min().item())
        max_val = float(vals.max().item())
    else:
        min_val = float("nan")
        max_val = float("nan")
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"finite={finite_count}/{total} nan={nan_count} inf={inf_count} "
        f"min={min_val} max={max_val}"
    )


def _nonfinite_row_details(name: str, inputs: tuple[Any, ...], tensor: torch.Tensor) -> str:
    if "embedding.word_embeddings" not in name or tensor.dim() < 3:
        return ""
    row_bad = ~torch.isfinite(tensor).all(dim=-1)
    bad_indices = row_bad.nonzero(as_tuple=False)
    if bad_indices.numel() == 0:
        return ""

    max_rows = int(os.environ.get("MILES_DEBUG_NAN_HOOK_MAX_ROWS", "32"))
    input_ids = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
    rows = []
    for raw_idx in bad_indices[:max_rows]:
        idx = [int(x.item()) for x in raw_idx]
        token = None
        if input_ids is not None and len(idx) <= input_ids.dim():
            try:
                token = int(input_ids[tuple(idx[: input_ids.dim()])].item())
            except Exception:
                token = None
        rows.append({"idx": idx, "token": token})
    return f" bad_rows={rows}"


def before_log_prob(args, model, store_prefix: str) -> None:
    pattern = os.environ.get("MILES_DEBUG_NAN_HOOK_PATTERN")
    max_modules = int(os.environ.get("MILES_DEBUG_NAN_HOOK_MAX_MODULES", "32"))
    rank = dist.get_rank() if dist.is_initialized() else 0
    log_path = Path(args.dump_details) / f"nan_hook_rank{rank}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    state = {"count": 0}

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")
            f.flush()
        print(message, flush=True)

    log(f"[nan_hook] rank={rank} store_prefix={store_prefix!r} pattern={pattern!r}")

    def make_hook(name: str):
        def hook(module, inputs, output):
            if state["count"] >= max_modules:
                return
            for tensor_name, tensor in _iter_tensors(output):
                if tensor.numel() == 0:
                    continue
                if not torch.isfinite(tensor).all():
                    state["count"] += 1
                    weight_stats = ""
                    weight = getattr(module, "weight", None)
                    if weight is not None and torch.is_tensor(weight):
                        weight_stats = f" weight_finite={bool(torch.isfinite(weight).all().item())}"
                    row_details = _nonfinite_row_details(name, inputs, tensor)
                    log(
                        f"[nan_hook] rank={rank} module={name} tensor={tensor_name} "
                        f"{_tensor_stats(tensor)}{weight_stats}{row_details}"
                    )
                    break

        return hook

    registered = 0
    for model_idx, model_module in enumerate(model):
        for name, module in model_module.named_modules():
            full_name = f"model[{model_idx}].{name}" if name else f"model[{model_idx}]"
            if pattern and pattern not in full_name:
                continue
            module.register_forward_hook(make_hook(full_name))
            registered += 1

    log(f"[nan_hook] rank={rank} registered={registered}")
