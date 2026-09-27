"""Export a completed CARL-D training report from a checkpoint."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.checkpointing import CHECKPOINT_TYPE, CHECKPOINT_VERSION
from utils.detection_metrics import DomainAP50History
from utils.reports_carld import write_training_report


def _training_summary(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    trainer_state = checkpoint.get("trainer_state")
    controller_state = checkpoint.get("controller")
    replay_state = checkpoint.get("replay_buffer")
    if not isinstance(trainer_state, Mapping):
        raise ValueError("Checkpoint does not contain CARL-D trainer state")
    if not isinstance(controller_state, Mapping):
        raise ValueError("Checkpoint does not contain CARL-D controller state")
    if not isinstance(replay_state, Mapping):
        raise ValueError("Checkpoint does not contain CARL-D replay state")

    raw_history = trainer_state.get("metric_history")
    if not isinstance(raw_history, Mapping):
        raise ValueError("Checkpoint does not contain metric history")
    history = DomainAP50History(num_domains=int(raw_history["num_domains"]))
    history.load_state_dict(raw_history)

    task_reports = [
        {key: value for key, value in dict(report).items() if key != "epochs"}
        for report in trainer_state.get("task_reports", ())
    ]
    action_history = list(controller_state.get(
        "events" if controller_state.get("algorithm") == "fixed_schedule" else "action_history",
        (),
    ))
    reward_history = list(controller_state.get("reward_history", ()))
    rewards_by_action: dict[str, list[float]] = {}
    for item in reward_history:
        action_name = str(item.get("action_name", "unknown"))
        rewards_by_action.setdefault(action_name, []).append(
            float(item.get("reward_total", 0.0))
        )
    mean_reward_by_action = {
        action_name: sum(values) / len(values)
        for action_name, values in rewards_by_action.items()
        if values
    }
    replay_domain_counts = Counter(
        int(record["domain_id"]) for record in replay_state.get("records", ())
    )
    model_parameters = (
        int(task_reports[-1]["model_parameters"])
        if task_reports
        else int(trainer_state["base_parameter_count"])
    )
    return {
        "benchmark": "CLAD-D",
        "completed_tasks": int(trainer_state["next_task_id"]),
        "optimizer_steps": int(trainer_state["optimizer_step"]),
        "controller_probe": history.summary(),
        "task_reports": task_reports,
        "action_distribution": dict(
            Counter(str(item.get("action_name", "unknown")) for item in action_history)
        ),
        "mean_reward_by_action": mean_reward_by_action,
        "controller_action_diagnostics": {
            "decisions": action_history,
            "rewards": reward_history,
        },
        "actions": list(trainer_state.get("action_events", ())),
        "replay_images": len(replay_state.get("records", ())),
        "replay_capacity": int(replay_state["current_capacity"]),
        "replay_domain_counts": dict(replay_domain_counts),
        "replay_loss_weight": float(
            trainer_state.get(
                "replay_loss_weight",
                checkpoint.get("config", {})
                .get("training", {})
                .get("replay_loss_weight", 1.0),
            )
        ),
        "replay_loss_weight_levels": list(
            checkpoint.get("config", {})
            .get("training", {})
            .get("replay_loss_weight_levels", ())
        ),
        "model_parameters": model_parameters,
        "base_model_parameters": int(trainer_state["base_parameter_count"]),
        "max_model_parameters": int(trainer_state["max_parameter_count"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export CARL-D training metrics directly from a final_*.pt or "
            "task checkpoint without loading the dataset or detector."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="CARL-D .pt checkpoint")
    parser.add_argument(
        "--output",
        help="Report path (default: timestamped results_*.txt beside the checkpoint)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CARL-D checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError("Checkpoint is not a CARL-D checkpoint")
    if int(checkpoint.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            "Unsupported CARL-D checkpoint version: "
            f"{checkpoint.get('checkpoint_version')}"
        )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (
            checkpoint_path.parent
            / f"results_{datetime.now().strftime('%d%m%Y_%H%M%S')}.txt"
        )
    )
    payload = {
        "run": "CARL-D training checkpoint export",
        "benchmark": "CLAD-D",
        "checkpoint": str(checkpoint_path),
        "checkpoint_version": int(checkpoint["checkpoint_version"]),
        "config": checkpoint.get("config", {}),
        "metrics": _training_summary(checkpoint),
    }
    write_training_report(output_path, payload)
    print(f"CARL-D training report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
