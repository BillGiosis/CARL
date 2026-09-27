"""Validation and reproducibility fingerprints for CARL-D configuration."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence

from models import (
    RESNET50_ARCHITECTURE,
    SUPPORTED_ARCHITECTURES,
)

FOREGROUND_CLASS_IDS = (1, 2, 3, 4, 5, 6)

_CONFIG_KEYS = {
    "": {
        "seed",
        "benchmark",
        "model",
        "training",
        "replay_buffer",
        "distillation",
        "resources",
        "data_loading",
        "controller",
        "reward",
        "evaluation",
        "checkpointing",
        "logging",
    },
    "benchmark": {
        "name",
        "data_root",
        "clad_repo_root",
        "num_tasks",
        "num_foreground_classes",
        "num_detector_classes",
        "background_class_id",
        "foreground_class_ids",
        "class_names",
        "domain_names",
    },
    "model": {
        "architecture",
        "weights",
        "num_classes",
        "min_size",
        "max_size",
        "trainable_backbone_layers",
        "box_score_thresh",
        "box_nms_thresh",
        "box_detections_per_img",
        "adapter_reduction",
        "preserve_coco_predictor_rows",
        "adapter_merge",
        "class_balanced_roi_loss",
        "adapters",
    },
    "training": {
        "batch_size",
        "eval_batch_size",
        "epochs_per_task",
        "optimizer",
        "learning_rate",
        "momentum",
        "weight_decay",
        "scheduler",
        "scheduler_milestones",
        "scheduler_gamma",
        "warmup_epochs",
        "warmup_start_factor",
        "replay_loss_weight",
        "replay_loss_weight_levels",
        "decision_frequency_epochs",
        "compile_dynamic",
        "cudnn_benchmark",
        "rare_image_sampling",
        "rare_image_sampling_strength",
        "class_balance_beta",
        "class_balance_max_weight",
        "stagewise_protection",
        "resnet_stage_lr_multipliers_after_task1",
    },
    "distillation": {
        "enabled",
        "loss_weight",
        "temperature",
        "max_proposals_per_image",
        "match_iou_threshold",
        "cache_dtype",
        "scope",
    },
    "replay_buffer": {
        "enabled",
        "initial_capacity",
        "max_capacity",
        "replay_batch_size",
        "refinement_candidate_count",
        "refinement_entropy_weight",
        "refinement_localization_weight",
        "refinement_rare_class_weight",
        "refinement_feature_redundancy_weight",
        "utility_sampling_strength",
        "decoded_image_cache_size",
        "decode_workers",
        "balanced_sampling",
        "class_balanced_admission",
        "batch_aware_sampling",
        "balanced_sampling_strength",
        "balanced_sampling_max_weight",
    },
    "resources": {"max_params_ratio"},
    "data_loading": {
        "num_workers",
        "pin_memory",
        "prefetch_factor",
        "persistent_workers",
    },
    "controller": {
        "algorithm",
        "learning_rate",
        "gamma",
        "tau",
        "replay_capacity",
        "batch_size",
        "learning_starts",
        "gradient_steps",
        "hidden_dims",
        "epsilon_start",
        "epsilon_end",
        "exploration_steps",
        "structural_expansion_cooldown_decisions",
        "feedback_ema_alpha",
        "feedback_support_scale",
        "freeze_cooldown_epochs",
        "replay_weight_freeze_epochs",
    },
    "reward": {
        "current_ap_gain",
        "old_ap_gain",
        "rare_ap_gain",
        "forgetting_delta",
        "parameter_delta",
        "invalid_action",
    },
    "evaluation": {"primary_metric", "iou_threshold", "foreground_class_ids"},
    "checkpointing": {"save_every_task", "final_path"},
    "logging": {"save_dir"},
}


# Fixed protocol values stay in effective checkpoint/report configurations.
# Legacy configs may repeat these values but cannot select different behavior.
_FIXED_RECIPE = {
    "benchmark": {
        "name": "CLAD-D", "num_tasks": 4, "num_foreground_classes": 6,
        "num_detector_classes": 7, "background_class_id": 0,
        "foreground_class_ids": [1, 2, 3, 4, 5, 6],
        "class_names": {1: "Pedestrian", 2: "Cyclist", 3: "Car", 4: "Truck",
                        5: "Tram (Bus)", 6: "Tricycle"},
        "domain_names": ["clear_day_citystreet", "day_highway", "night", "rainy_day"],
    },
    "model": {
        "architecture": "fasterrcnn_resnet50_fpn_v2", "num_classes": 7,
        "trainable_backbone_layers": 5, "adapter_merge": "parallel",
        "preserve_coco_predictor_rows": True,
    },
    "training": {"optimizer": "sgd", "scheduler": "multistep"},
    "distillation": {"scope": "replay_roi_logits", "cache_dtype": "float16"},
    "replay_buffer": {"initial_capacity": 250, "max_capacity": 250},
    "evaluation": {"primary_metric": "domain_macro_map50", "iou_threshold": 0.5,
                   "foreground_class_ids": [1, 2, 3, 4, 5, 6]},
}


def normalize_carl_d_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve replay-dependent KD without mutating the caller's recipe."""
    result = deepcopy(dict(config))
    for section, values in _FIXED_RECIPE.items():
        target = result.setdefault(section, {})
        if not isinstance(target, dict):
            raise TypeError(f"{section} must be a mapping")
        for key, expected in values.items():
            if key in target and target[key] != expected:
                raise ValueError(
                    f"{section}.{key} is fixed to {expected!r}; "
                    "this configuration belongs to an unsupported recipe"
                )
            target[key] = deepcopy(expected)
    replay_enabled = result.get("replay_buffer", {}).get("enabled", True)
    kd_enabled = result.get("distillation", {}).get("enabled", False)
    if not isinstance(replay_enabled, bool):
        raise TypeError("replay_buffer.enabled must be boolean")
    if not isinstance(kd_enabled, bool):
        raise TypeError("distillation.enabled must be boolean")
    if not replay_enabled:
        result.setdefault("distillation", {})["enabled"] = False
    return result


def validate_carl_d_config(config: Mapping[str, Any]) -> None:
    """Reject misspelled or contradictory fixed-method configuration."""
    config = normalize_carl_d_config(config)
    for section, allowed in _CONFIG_KEYS.items():
        mapping = config if not section else config.get(section, {})
        if not isinstance(mapping, Mapping):
            label = "top-level config" if not section else section
            raise TypeError(f"CARL-D {label} must be a mapping")
        unknown = set(mapping) - allowed
        if unknown:
            label = "top-level" if not section else section
            raise ValueError(f"Unknown CARL-D {label} keys: {sorted(unknown)}")

    controller = config.get("controller", {})
    if controller.get("algorithm", "dqn") not in ("dqn", "fixed_schedule"):
        raise ValueError("controller.algorithm must be 'dqn' or 'fixed_schedule'")
    if (
        controller.get("algorithm") == "fixed_schedule"
        and config.get("training", {}).get("decision_frequency_epochs", 2) != 2
    ):
        raise ValueError("CARL-FS requires the unchanged two-epoch evaluation interval")
    evaluation = config.get("evaluation", {})
    if evaluation.get("primary_metric", "domain_macro_map50") != "domain_macro_map50":
        raise ValueError("CARL-D primary metric must be domain_macro_map50")
    if float(evaluation.get("iou_threshold", 0.5)) != 0.5:
        raise ValueError("CARL-D evaluation.iou_threshold must be 0.5")
    if (
        tuple(evaluation.get("foreground_class_ids", FOREGROUND_CLASS_IDS))
        != FOREGROUND_CLASS_IDS
    ):
        raise ValueError("CARL-D metrics require foreground class IDs 1..6")
    adapters = config.get("model", {}).get("adapters", {})
    if not isinstance(adapters, Mapping):
        raise TypeError("CARL-D model.adapters must be a mapping")
    unknown_adapters = set(adapters) - {"enabled", "groups", "max_generations"}
    if unknown_adapters:
        raise ValueError(
            f"Unknown CARL-D model.adapters keys: {sorted(unknown_adapters)}"
        )
    preserve_rows = config.get("model", {}).get("preserve_coco_predictor_rows", True)
    if not isinstance(preserve_rows, bool):
        raise TypeError("model.preserve_coco_predictor_rows must be boolean")
    class_balanced = config.get("model", {}).get("class_balanced_roi_loss", True)
    if not isinstance(class_balanced, bool):
        raise TypeError("model.class_balanced_roi_loss must be boolean")
    model_config = config.get("model", {})
    architecture = str(model_config.get("architecture", RESNET50_ARCHITECTURE)).lower()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"model.architecture must be one of {SUPPORTED_ARCHITECTURES}")
    layers = model_config.get("trainable_backbone_layers", 3)
    if isinstance(layers, bool) or not isinstance(layers, int) or not 0 <= layers <= 5:
        raise ValueError("model.trainable_backbone_layers must be an integer in [0, 5]")
    max_generations = adapters.get(
        "max_generations", {"backbone": 3, "fpn": 3, "roi": 4}
    )
    if not isinstance(max_generations, Mapping):
        raise TypeError("model.adapters.max_generations must be a mapping")
    if set(max_generations) != {"backbone", "fpn", "roi"}:
        raise ValueError(
            "model.adapters.max_generations must define backbone, fpn, and roi"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in max_generations.values()
    ):
        raise TypeError("adapter generation limits must be nonnegative integers")

    training = config.get("training", {})
    for key in ("batch_size", "eval_batch_size"):
        value = training.get(key, 2)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TypeError(f"training.{key} must be a positive integer")
    epochs = int(training.get("epochs_per_task", 50))
    interval = int(training.get("decision_frequency_epochs", 2))
    if epochs <= 0 or interval <= 0 or epochs % interval:
        raise ValueError(
            "training.epochs_per_task must be a positive multiple of "
            "training.decision_frequency_epochs"
        )

    gamma = float(training.get("scheduler_gamma", 0.1))
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("training.scheduler_gamma must be finite and positive")
    raw_milestones = training.get("scheduler_milestones", (34, 44))
    if isinstance(raw_milestones, (str, bytes)) or not isinstance(
        raw_milestones, Sequence
    ):
        raise TypeError("training.scheduler_milestones must be a sequence")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_milestones
    ):
        raise TypeError("training.scheduler_milestones must contain integers")
    milestones = tuple(raw_milestones)
    if (
        not milestones
        or tuple(sorted(set(milestones))) != milestones
        or any(value <= 0 or value >= epochs for value in milestones)
    ):
        raise ValueError(
            "training.scheduler_milestones must be unique, increasing, and "
            "strictly inside the task epoch range"
        )
    if any(value % interval for value in milestones):
        raise ValueError(
            "training.scheduler_milestones must align with controller "
            "decision boundaries"
        )

    raw_warmup_epochs = training.get("warmup_epochs", 0)
    if isinstance(raw_warmup_epochs, bool) or not isinstance(raw_warmup_epochs, int):
        raise TypeError("training.warmup_epochs must be an integer")
    warmup_epochs = raw_warmup_epochs
    warmup_start_factor = float(training.get("warmup_start_factor", 1e-3))
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError(
            "training.warmup_epochs must be nonnegative and smaller than "
            "training.epochs_per_task"
        )
    if (
        not math.isfinite(warmup_start_factor)
        or warmup_start_factor <= 0.0
        or warmup_start_factor > 1.0
    ):
        raise ValueError("training.warmup_start_factor must be in the interval (0, 1]")
    if warmup_epochs >= milestones[0]:
        raise ValueError(
            "training warmup must finish before the first scheduler milestone"
        )

    compile_dynamic = training.get("compile_dynamic", True)
    if compile_dynamic is not None and not isinstance(compile_dynamic, bool):
        raise TypeError("training.compile_dynamic must be boolean or null")
    cudnn_benchmark = training.get("cudnn_benchmark", False)
    if not isinstance(cudnn_benchmark, bool):
        raise TypeError("training.cudnn_benchmark must be a boolean")
    rare_sampling = training.get("rare_image_sampling", True)
    if not isinstance(rare_sampling, bool):
        raise TypeError("training.rare_image_sampling must be boolean")
    sampling_strength = float(training.get("rare_image_sampling_strength", 0.5))
    if not math.isfinite(sampling_strength) or not 0.0 <= sampling_strength <= 1.0:
        raise ValueError("training.rare_image_sampling_strength must be in [0, 1]")
    balance_beta = float(training.get("class_balance_beta", 0.999))
    if not math.isfinite(balance_beta) or not 0.0 <= balance_beta < 1.0:
        raise ValueError("training.class_balance_beta must be in [0, 1)")
    maximum_class_weight = float(training.get("class_balance_max_weight", 3.0))
    if not math.isfinite(maximum_class_weight) or maximum_class_weight < 1.0:
        raise ValueError(
            "training.class_balance_max_weight must be finite and at least one"
        )
    stagewise_protection = training.get("stagewise_protection", False)
    if not isinstance(stagewise_protection, bool):
        raise TypeError("training.stagewise_protection must be boolean")
    stage_key = "resnet_stage_lr_multipliers_after_task1"
    raw_stage_multipliers = training.get(stage_key, (0.0, 0.0, 0.05, 0.25, 0.5))
    if isinstance(raw_stage_multipliers, (str, bytes)) or not isinstance(
        raw_stage_multipliers, Sequence
    ):
        raise TypeError(f"training.{stage_key} must be a sequence")
    stage_multipliers = tuple(float(value) for value in raw_stage_multipliers)
    if len(stage_multipliers) != 5 or any(
        not math.isfinite(value) or value < 0.0 for value in stage_multipliers
    ):
        raise ValueError(
            f"training.{stage_key} must contain 5 finite nonnegative values"
        )
    if stagewise_protection and layers != 5:
        raise ValueError(
            "ResNet stagewise protection requires trainable_backbone_layers=5"
        )

    distillation = config.get("distillation", {})
    distillation_enabled = distillation.get("enabled", False)
    if not isinstance(distillation_enabled, bool):
        raise TypeError("distillation.enabled must be boolean")
    distillation_weight = float(distillation.get("loss_weight", 0.5))
    temperature = float(distillation.get("temperature", 2.0))
    match_iou = float(distillation.get("match_iou_threshold", 0.5))
    max_proposals = distillation.get("max_proposals_per_image", 256)
    if not math.isfinite(distillation_weight) or distillation_weight < 0.0:
        raise ValueError("distillation.loss_weight must be finite and nonnegative")
    if distillation_enabled and distillation_weight <= 0.0:
        raise ValueError(
            "distillation.loss_weight must be positive when distillation is enabled"
        )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("distillation.temperature must be finite and positive")
    if not math.isfinite(match_iou) or not 0.0 <= match_iou <= 1.0:
        raise ValueError("distillation.match_iou_threshold must be in [0, 1]")
    if (
        isinstance(max_proposals, bool)
        or not isinstance(max_proposals, int)
        or max_proposals <= 0
    ):
        raise TypeError(
            "distillation.max_proposals_per_image must be a positive integer"
        )
    scope = str(distillation.get("scope", "replay_roi_logits"))
    if scope != "replay_roi_logits":
        raise ValueError("distillation.scope must be replay_roi_logits")
    replay_weight = float(training.get("replay_loss_weight", 1.0))
    raw_replay_levels = training.get("replay_loss_weight_levels", (0.5, 1.0, 2.0))
    if isinstance(raw_replay_levels, (str, bytes)) or not isinstance(
        raw_replay_levels, Sequence
    ):
        raise TypeError("training.replay_loss_weight_levels must be a sequence")
    replay_levels = tuple(float(value) for value in raw_replay_levels)
    if (
        len(replay_levels) < 2
        or any(not math.isfinite(value) or value <= 0.0 for value in replay_levels)
        or tuple(sorted(set(replay_levels))) != replay_levels
    ):
        raise ValueError(
            "training.replay_loss_weight_levels must contain at least two "
            "unique, increasing, finite positive values"
        )
    if not any(math.isclose(replay_weight, value) for value in replay_levels):
        raise ValueError(
            "training.replay_loss_weight must be one of "
            "training.replay_loss_weight_levels"
        )

    replay = config.get("replay_buffer", {})
    for key in ("decoded_image_cache_size", "decode_workers"):
        value = replay.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(f"replay_buffer.{key} must be a nonnegative integer")
    initial_capacity = int(replay.get("initial_capacity", 250))
    maximum_capacity = int(replay.get("max_capacity", 250))
    if initial_capacity != 250 or maximum_capacity != 250:
        raise ValueError(
            "primary CARL-D uses the fixed official 250-image replay budget; "
            "initial_capacity and max_capacity must both equal 250"
        )
    if not isinstance(replay.get("balanced_sampling", True), bool):
        raise TypeError("replay_buffer.balanced_sampling must be boolean")
    if not isinstance(replay.get("class_balanced_admission", True), bool):
        raise TypeError("replay_buffer.class_balanced_admission must be boolean")
    if not isinstance(replay.get("batch_aware_sampling", True), bool):
        raise TypeError("replay_buffer.batch_aware_sampling must be boolean")
    balance_strength = float(replay.get("balanced_sampling_strength", 0.5))
    if not math.isfinite(balance_strength) or not 0.0 <= balance_strength <= 1.0:
        raise ValueError("replay_buffer.balanced_sampling_strength must be in [0, 1]")
    replay_max_weight = float(replay.get("balanced_sampling_max_weight", 3.0))
    if not math.isfinite(replay_max_weight) or replay_max_weight < 1.0:
        raise ValueError(
            "replay_buffer.balanced_sampling_max_weight must be at least one"
        )
    utility_sampling_strength = float(replay.get("utility_sampling_strength", 0.25))
    if (
        not math.isfinite(utility_sampling_strength)
        or not 0.0 <= utility_sampling_strength <= 1.0
    ):
        raise ValueError("replay_buffer.utility_sampling_strength must be in [0, 1]")
    cooldown = config.get("controller", {}).get(
        "structural_expansion_cooldown_decisions", 1
    )
    if isinstance(cooldown, bool) or not isinstance(cooldown, int) or cooldown < 0:
        raise TypeError(
            "controller.structural_expansion_cooldown_decisions must be a "
            "nonnegative integer"
        )
    feedback_alpha = float(config.get("controller", {}).get("feedback_ema_alpha", 0.3))
    if not math.isfinite(feedback_alpha) or not 0.0 < feedback_alpha <= 1.0:
        raise ValueError("controller.feedback_ema_alpha must be in (0, 1]")
    support_scale = float(config.get("controller", {}).get("feedback_support_scale", 20.0))
    if not math.isfinite(support_scale) or support_scale <= 0:
        raise ValueError("controller.feedback_support_scale must be finite and positive")
    for name, default in (
        ("freeze_cooldown_epochs", 2),
        ("replay_weight_freeze_epochs", 4),
    ):
        value = config.get("controller", {}).get(name, default)
        interval = int(training.get("decision_frequency_epochs", 2))
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value % interval
        ):
            raise ValueError(
                f"controller.{name} must be a non-negative integer multiple of "
                "training.decision_frequency_epochs"
            )


def _dataset_ids(dataset: Any) -> Sequence[Any]:
    official = getattr(dataset, "official_dataset", dataset)
    ids = getattr(official, "ids", None)
    if ids is None:
        return tuple(range(len(dataset)))
    return tuple(ids)


def dataset_split_fingerprint(benchmark: Any) -> str:
    """Hash the ordered train/validation domain IDs used by this run."""
    payload: Dict[str, Any] = {
        "benchmark": "CLAD-D",
        "train": [],
        "validation": [],
    }
    for key, datasets in (
        ("train", benchmark.train_datasets),
        ("validation", benchmark.validation_datasets),
    ):
        for domain_id, dataset in enumerate(datasets):
            payload[key].append(
                {
                    "domain_id": domain_id,
                    "ids": [str(image_id) for image_id in _dataset_ids(dataset)],
                }
            )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stringify_mapping_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stringify_mapping_keys(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_mapping_keys(item) for item in value]
    return value


def resume_config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash settings that affect resumed model/controller trajectories."""
    config = normalize_carl_d_config(config)
    benchmark = dict(config.get("benchmark", {}))
    benchmark.pop("data_root", None)
    benchmark.pop("clad_repo_root", None)
    model = dict(config.get("model", {}))
    # Initial weights are irrelevant once checkpoint detector weights are read.
    model.pop("weights", None)
    training = dict(config.get("training", {}))
    critical = {
        "seed": config.get("seed", 0),
        "benchmark": benchmark,
        "model": model,
        "training": training,
        "replay_buffer": config.get("replay_buffer", {}),
        "resources": config.get("resources", {}),
        "data_loading": config.get("data_loading", {}),
        "controller": config.get("controller", {}),
        "reward": config.get("reward", {}),
        "evaluation": config.get("evaluation", {}),
    }
    encoded = json.dumps(
        _stringify_mapping_keys(critical),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "dataset_split_fingerprint",
    "resume_config_fingerprint",
    "validate_carl_d_config",
]
