"""Versioned checkpoint and exact-resume utilities for CARL-D."""

from __future__ import annotations

import os
import random
import tempfile
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

CHECKPOINT_TYPE = "CARL-D"
CHECKPOINT_VERSION = 2


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _architecture_manifest(model: torch.nn.Module) -> Dict[str, Any]:
    method = getattr(model, "architecture_manifest", None)
    return deepcopy(method()) if callable(method) else {}


def _apply_architecture_manifest(
    model: torch.nn.Module, manifest: Mapping[str, Any]
) -> None:
    if not manifest:
        return
    for method_name in ("apply_architecture_manifest", "load_architecture_manifest"):
        method = getattr(model, method_name, None)
        if callable(method):
            method(deepcopy(dict(manifest)))
            return
    raise ValueError(
        "Checkpoint contains a dynamic architecture manifest but the model "
        "cannot reconstruct it"
    )


def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    replay_buffer: Any,
    controller: Any,
    trainer_state: Mapping[str, Any],
    config: Mapping[str, Any],
    dataset_fingerprint: str,
) -> Dict[str, Any]:
    return {
        "checkpoint_type": CHECKPOINT_TYPE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture_manifest": _architecture_manifest(model),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "replay_buffer": replay_buffer.state_dict(),
        "controller": controller.state_dict(),
        "trainer_state": deepcopy(dict(trainer_state)),
        "config": deepcopy(dict(config)),
        "dataset_fingerprint": str(dataset_fingerprint),
        "rng_state": capture_rng_state(),
    }


def save_checkpoint(path: str, **components: Any) -> None:
    """Atomically write a complete CARL-D checkpoint."""
    checkpoint = build_checkpoint(**components)
    absolute_path = os.path.abspath(path)
    directory = os.path.dirname(absolute_path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=directory,
            prefix=".carl_d_checkpoint_",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, absolute_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    replay_buffer: Optional[Any] = None,
    controller: Optional[Any] = None,
    expected_dataset_fingerprint: Optional[str] = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return restore_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        replay_buffer=replay_buffer,
        controller=controller,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
        restore_rng=restore_rng,
    )


def restore_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    replay_buffer: Optional[Any] = None,
    controller: Optional[Any] = None,
    expected_dataset_fingerprint: Optional[str] = None,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    """Restore a checkpoint mapping that has already been loaded once."""
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError(
            "Checkpoint is not a CARL-D checkpoint; incompatible legacy "
            "framework checkpoints cannot be loaded"
        )
    if int(checkpoint.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported CARL-D checkpoint version: "
            f"{checkpoint.get('checkpoint_version')}"
        )
    fingerprint = str(checkpoint.get("dataset_fingerprint", ""))
    if expected_dataset_fingerprint is not None and fingerprint != str(
        expected_dataset_fingerprint
    ):
        raise ValueError("Dataset split fingerprint differs from checkpoint")

    _apply_architecture_manifest(model, checkpoint["architecture_manifest"])
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint["optimizer"] is not None:
        # Stage metadata belongs to the current model, not historical key names.
        stages = [group.get("carl_backbone_stage") for group in optimizer.param_groups]
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group, stage in zip(optimizer.param_groups, stages):
            group["carl_backbone_stage"] = stage
    if scheduler is not None and checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if replay_buffer is not None:
        replay_buffer.load_state_dict(checkpoint["replay_buffer"])
    if controller is not None:
        controller.load_state_dict(checkpoint["controller"])
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return {
        "trainer_state": checkpoint["trainer_state"],
        "config": checkpoint["config"],
        "dataset_fingerprint": fingerprint,
    }


__all__ = [
    "CHECKPOINT_TYPE",
    "CHECKPOINT_VERSION",
    "build_checkpoint",
    "capture_rng_state",
    "load_checkpoint",
    "restore_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
]
