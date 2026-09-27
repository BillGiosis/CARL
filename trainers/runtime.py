"""Tensor routing, initialization and health checks used by the trainer."""

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_grad_scaler(enabled: bool):
    scaler_type = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if scaler_type is not None:
        return scaler_type("cuda", enabled=enabled)
    # Compatibility with the oldest PyTorch version allowed by requirements.
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _finite_or_zero(value: Any) -> float:
    number = float(value)
    return number if math.isfinite(number) else 0.0


def _box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Small dependency-free IoU helper for replay scoring and test fallbacks."""
    boxes1 = torch.as_tensor(boxes1, dtype=torch.float32)
    boxes2 = torch.as_tensor(boxes2, dtype=torch.float32)
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((len(boxes1), len(boxes2)))
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=2)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(dim=1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(dim=1)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=torch.finfo(intersection.dtype).eps)


def _to_device(value: Any, device: torch.device) -> Any:
    return tree_map(
        lambda item: (
            item.to(device=device, non_blocking=True) if torch.is_tensor(item) else item
        ),
        value,
    )


def _cpu_detection_dict(value: Mapping[str, Any]) -> Dict[str, Any]:
    def to_cpu_leaf(item: Any) -> Any:
        if torch.is_tensor(item):
            detached = item.detach()
            # A device-to-CPU transfer already owns independent storage. Only
            # CPU inputs need an explicit clone to preserve that contract.
            return (
                detached.clone()
                if detached.device.type == "cpu"
                else detached.to(device="cpu")
            )
        return deepcopy(item)

    return tree_map(to_cpu_leaf, dict(value))


def _cpu_detection_batch(outputs: Any) -> Any:
    """Copy tensor leaves in dtype/device groups, preserving values and order."""
    leaves, spec = tree_flatten(outputs)
    copied = [None] * len(leaves)
    groups = {}
    for index, leaf in enumerate(leaves):
        if not torch.is_tensor(leaf):
            copied[index] = deepcopy(leaf)
        elif leaf.device.type == "cpu":
            copied[index] = leaf.detach().clone()
        else:
            groups.setdefault((leaf.device, leaf.dtype), []).append(index)
    for indices in groups.values():
        packed = torch.cat([leaves[i].detach().reshape(-1) for i in indices]).cpu()
        offset = 0
        for i in indices:
            count = leaves[i].numel()
            copied[i] = packed[offset : offset + count].reshape(leaves[i].shape)
            offset += count
    return tree_unflatten(copied, spec)


class _CudaReplayTransfer:
    """Stage an already-augmented CPU batch; never sample or consume RNG here."""

    def __init__(self, batch: Dict[str, Any], device: torch.device, stream):
        def pin(tensor):
            if not torch.is_tensor(tensor):
                return tensor
            if tensor.device.type != "cpu":
                raise ValueError("Replay transfer requires a CPU source batch")
            return tensor if tensor.is_pinned() else tensor.pin_memory()

        self.source = tree_map(pin, (batch["images"], batch["targets"]))
        self.device = device
        with torch.cuda.stream(stream):
            images, targets = _to_device(self.source, device)
            self.ready = torch.cuda.Event()
            self.ready.record(stream)
        self.batch = {**batch, "images": images, "targets": targets}

    def wait(self) -> Dict[str, Any]:
        consumer = torch.cuda.current_stream(self.device)
        consumer.wait_event(self.ready)

        def record(tensor):
            if torch.is_tensor(tensor) and tensor.device.type == "cuda":
                tensor.record_stream(consumer)
            return tensor

        tree_map(record, (self.batch["images"], self.batch["targets"]))
        return self.batch


def check_model_health(model: nn.Module) -> None:
    """Fail immediately if a detector parameter contains NaN or Inf."""
    invalid = [
        name
        for name, parameter in model.named_parameters()
        if not torch.isfinite(parameter).all()
    ]
    if invalid:
        preview = ", ".join(invalid[:5])
        suffix = " ..." if len(invalid) > 5 else ""
        raise FloatingPointError(f"Non-finite detector parameters: {preview}{suffix}")
