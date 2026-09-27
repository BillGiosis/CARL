from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "carl_d.yaml"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.reports_carld import write_training_report


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"CARL-D config must be a YAML mapping: {config_path}")
    benchmark = config.setdefault("benchmark", {})
    benchmark.setdefault("name", "CLAD-D")
    if benchmark.get("name") != "CLAD-D":
        raise ValueError("train_carl_d.py accepts only a CLAD-D configuration")
    return config


def _timestamped_path(path: Path, timestamp: str) -> Path:
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def _default_checkpoint(config: Mapping[str, Any], timestamp: str) -> Path:
    configured = config.get("checkpointing", {}).get("final_path")
    if configured:
        return _timestamped_path(Path(str(configured)).expanduser(), timestamp)
    save_dir = config.get("logging", {}).get(
        "save_dir", REPOSITORY_ROOT / "checkpoints" / "carl_d"
    )
    return Path(str(save_dir)).expanduser() / f"final_{timestamp}.pt"


def _default_output(checkpoint_path: Path, timestamp: str) -> Path:
    return checkpoint_path.parent / f"results_{timestamp}.txt"


def _compact_training_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Remove epoch detail while retaining task-level results."""
    compact = dict(result)
    compact["task_reports"] = [
        {key: value for key, value in dict(report).items() if key != "epochs"}
        for report in result.get("task_reports", ())
    ]
    return compact


def _write_report(path: str, payload: Mapping[str, Any]) -> None:
    output_path = write_training_report(path, payload)
    print(f"CARL-D training report written to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train detection-only CARL-D with a shared Faster R-CNN detector "
            "sequentially on the four CLAD-D domains."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="CARL-D YAML configuration (default: config/carl_d.yaml)",
    )
    parser.add_argument(
        "--resume",
        help="Resume the complete training state from a CARL-D checkpoint",
    )
    parser.add_argument(
        "--device",
        help="Torch device such as cuda, cuda:0, or cpu (trainer default if omitted)",
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "Exact final-checkpoint path override (default: timestamped "
            "final_*.pt under logging.save_dir)"
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Exact training-report path override, including the final "
            "test AP50 metrics (default: timestamped results_*.txt "
            "beside the final checkpoint)"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    # Keep CLI help independent of trainer initialization.
    from trainers import CARLDTrainer

    trainer = CARLDTrainer.from_config(
        config,
        device=args.device,
        load_pretrained_weights=not bool(args.resume),
    )
    if args.resume:
        trainer.load_checkpoint(args.resume, load_training_state=True)

    training_started = time.perf_counter()
    training_result = trainer.train()
    training_time_seconds = time.perf_counter() - training_started
    checkpoint_path = (
        Path(args.checkpoint).expanduser()
        if args.checkpoint
        else _default_checkpoint(config, trainer.run_timestamp)
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(checkpoint_path))

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else _default_output(checkpoint_path, trainer.run_timestamp)
    )
    report_payload = {
        "run": "CARL-D training",
        "benchmark": "CLAD-D",
        "config_path": str(config_path),
        "config": trainer.config,
        "device": args.device,
        "resumed_from": args.resume,
        "checkpoint": str(checkpoint_path),
        "training_time_seconds": training_time_seconds,
        "test_evaluation_time_seconds": None,
        "metrics": _compact_training_result(training_result),
        "test_metrics": {},
    }
    # Preserve completed training metrics even if the separate final-test pass
    # encounters an annotation, I/O, or evaluator failure. A successful test
    # pass rewrites this same report with its final metrics below.
    _write_report(str(output_path), report_payload)

    test_started = time.perf_counter()
    test_result = trainer.evaluate_test()
    test_evaluation_time_seconds = time.perf_counter() - test_started
    report_payload["test_evaluation_time_seconds"] = test_evaluation_time_seconds
    report_payload["test_metrics"] = test_result
    _write_report(str(output_path), report_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
