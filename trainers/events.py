"""Serializable action-execution records for training and checkpoint reports."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass
class ActionExecutionEvent:
    """Typed result of applying one controller action to the trainer."""

    task_id: int
    applies_to_task_id: int
    epoch: int
    optimizer_step: int
    action_id: int
    action_name: str
    replay_loss_weight_before: float
    replay_before: Dict[str, Any]
    changed: bool = False
    invalid: bool = False
    replay_loss_weight_after: float = 0.0
    replay_after: Dict[str, Any] = field(default_factory=dict)
    candidate_count: Optional[int] = None
    duration_epochs: Optional[int] = None
    until_task_epoch: Optional[int] = None
    added_parameters: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "applies_to_task_id": self.applies_to_task_id,
            "epoch": self.epoch,
            "optimizer_step": self.optimizer_step,
            "action_id": self.action_id,
            "action_name": self.action_name,
            "changed": self.changed,
            "invalid": self.invalid,
            "replay_loss_weight_before": self.replay_loss_weight_before,
            "replay_before": deepcopy(self.replay_before),
            "replay_loss_weight_after": self.replay_loss_weight_after,
            "replay_after": deepcopy(self.replay_after),
        }
        for name in (
            "candidate_count",
            "duration_epochs",
            "until_task_epoch",
            "added_parameters",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ActionExecutionEvent":
        return cls(
            task_id=int(payload["task_id"]),
            applies_to_task_id=int(payload["applies_to_task_id"]),
            epoch=int(payload["epoch"]),
            optimizer_step=int(payload["optimizer_step"]),
            action_id=int(payload["action_id"]),
            action_name=str(payload["action_name"]),
            changed=bool(payload.get("changed", False)),
            invalid=bool(payload.get("invalid", False)),
            replay_loss_weight_before=float(payload["replay_loss_weight_before"]),
            replay_before=deepcopy(dict(payload["replay_before"])),
            replay_loss_weight_after=float(
                payload.get("replay_loss_weight_after", 0.0)
            ),
            replay_after=deepcopy(dict(payload.get("replay_after", {}))),
            candidate_count=(
                None
                if payload.get("candidate_count") is None
                else int(payload["candidate_count"])
            ),
            duration_epochs=(
                None
                if payload.get("duration_epochs") is None
                else int(payload["duration_epochs"])
            ),
            until_task_epoch=(
                None
                if payload.get("until_task_epoch") is None
                else int(payload["until_task_epoch"])
            ),
            added_parameters=(
                None
                if payload.get("added_parameters") is None
                else int(payload["added_parameters"])
            ),
        )
