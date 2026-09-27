from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "carl_d.yaml"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.checkpointing import CHECKPOINT_TYPE, CHECKPOINT_VERSION
from utils.reports_carld import write_test_report


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"CARL-D config must be a YAML mapping: {config_path}")
    config.setdefault("benchmark", {}).setdefault("name", "CLAD-D")
    if config["benchmark"]["name"] != "CLAD-D":
        raise ValueError("evaluate_carl_d.py accepts only a CLAD-D configuration")
    return config


def _latest_timestamped_checkpoint(path: Path) -> Path:
    candidates = list(path.parent.glob(f"{path.stem}_*{path.suffix}"))
    if not candidates:
        return path
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)


def _default_checkpoint(config: Mapping[str, Any]) -> Path:
    configured = config.get("checkpointing", {}).get("final_path")
    if configured:
        return _latest_timestamped_checkpoint(Path(str(configured)).expanduser())
    save_dir = config.get("logging", {}).get(
        "save_dir", REPOSITORY_ROOT / "checkpoints" / "carl_d"
    )
    return _latest_timestamped_checkpoint(Path(str(save_dir)).expanduser() / "final.pt")


def _default_output(checkpoint_path: Path, timestamp: str) -> Path:
    return checkpoint_path.parent / f"test_results_{timestamp}.txt"


def _write_report(path: str, payload: Mapping[str, Any]) -> None:
    output_path = write_test_report(path, payload)
    print(f"CARL-D test report written to {output_path}")


def _config_for_checkpoint(
    runtime_config: Mapping[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    """Use the saved model contract while retaining runtime data settings."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError("Checkpoint belongs to an incompatible framework")
    if int(checkpoint.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            "Unsupported CARL-D checkpoint version: "
            f"{checkpoint.get('checkpoint_version')}"
        )

    saved_config = checkpoint.get("config")
    manifest = checkpoint.get("architecture_manifest")
    if not isinstance(saved_config, Mapping):
        raise ValueError("Checkpoint does not contain a CARL-D configuration")
    if not isinstance(manifest, Mapping):
        raise ValueError("Checkpoint does not contain an architecture manifest")
    saved_model = saved_config.get("model")
    if not isinstance(saved_model, Mapping):
        raise ValueError("Checkpoint configuration does not contain a model section")

    model_config = deepcopy(dict(saved_model))
    # Ignore training-only sampler metadata when loading a checkpoint for inference.
    model_config.pop("class_balanced_roi_sampling", None)
    manifest_architecture = manifest.get("architecture")
    if not isinstance(manifest_architecture, str) or not manifest_architecture:
        raise ValueError("Checkpoint architecture manifest has no architecture")
    manifest_architecture = manifest_architecture.lower()
    configured_architecture = model_config.get("architecture")
    if configured_architecture is None:
        model_config["architecture"] = manifest_architecture
    elif str(configured_architecture).lower() != manifest_architecture:
        raise ValueError(
            "Checkpoint model configuration and architecture manifest disagree"
        )
    else:
        model_config["architecture"] = manifest_architecture
    if "num_classes" not in manifest:
        raise ValueError("Checkpoint architecture manifest has no class count")
    model_config.setdefault("num_classes", int(manifest["num_classes"]))
    if "num_classes" in model_config and int(model_config["num_classes"]) != int(
        manifest["num_classes"]
    ):
        raise ValueError(
            "Checkpoint model configuration and manifest class counts disagree"
        )

    effective_config = deepcopy(dict(runtime_config))
    # Detection thresholds, transforms, head type, and architecture all affect
    # metrics, so the complete saved model section is authoritative. Dataset
    # paths, loader settings, and evaluation batch size remain runtime choices.
    effective_config["model"] = model_config
    effective_config.setdefault("controller", {})["algorithm"] = (
        saved_config.get("controller", {}).get("algorithm", "dqn")
    )
    return effective_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen CARL-D checkpoint once on all CLAD-D test "
            "domains and export COCO 101-point AP50 with per-class, "
            "official-domain and environmental breakdowns."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="CARL-D YAML configuration (default: config/carl_d.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "Checkpoint to evaluate (default: most recent timestamped "
            "final_*.pt under logging.save_dir)"
        ),
    )
    parser.add_argument(
        "--device",
        help="Torch device such as cuda, cuda:0, or cpu (trainer default if omitted)",
    )
    parser.add_argument(
        "--output",
        help=(
            "Exact test-report path override (default: "
            "timestamped test_results_*.txt beside the checkpoint)"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    checkpoint_path = (
        Path(args.checkpoint).expanduser()
        if args.checkpoint
        else _default_checkpoint(config)
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CARL-D checkpoint does not exist: {checkpoint_path}")

    effective_config = _config_for_checkpoint(config, checkpoint_path)

    from trainers import CARLDTrainer

    trainer = CARLDTrainer.from_config(
        effective_config,
        device=args.device,
        load_pretrained_weights=False,
    )
    trainer.load_checkpoint(str(checkpoint_path), load_training_state=False)
    test_result = trainer.evaluate_test()

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else _default_output(
            checkpoint_path,
            datetime.now().strftime("%d%m%Y_%H%M%S"),
        )
    )
    _write_report(
        str(output_path),
        {
            "run": "CARL-D final evaluation",
            "benchmark": "CLAD-D",
            "config_path": str(config_path),
            "config": trainer.config,
            "device": args.device,
            "checkpoint": str(checkpoint_path),
            "metrics": test_result,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
