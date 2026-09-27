"""Human-readable CARL-D training and evaluation reports."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

SEPARATOR = "=" * 80



def _value(mapping: Mapping[Any, Any], key: int | str, default: Any = None) -> Any:
    if key in mapping:
        return mapping[key]
    alternate = str(key) if isinstance(key, int) else int(key) if key.isdigit() else key
    return mapping.get(alternate, default)


def _metric(value: Any) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    return f"{number:.4f}" if math.isfinite(number) else "N/A"


def _timestamp_for(path: Path) -> str:
    stem_parts = path.stem.rsplit("_", 2)
    if len(stem_parts) >= 3:
        raw = "_".join(stem_parts[-2:])
        try:
            return datetime.strptime(raw, "%d%m%Y_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _config(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = payload.get("config", {})
    return value if isinstance(value, Mapping) else {}


def _class_names(config: Mapping[str, Any]) -> dict[int, str]:
    configured = config.get("benchmark", {}).get("class_names", {})
    defaults = {
        1: "Pedestrian",
        2: "Cyclist",
        3: "Car",
        4: "Truck",
        5: "Tram (Bus)",
        6: "Tricycle",
    }
    return {
        class_id: str(_value(configured, class_id, name))
        for class_id, name in defaults.items()
    }


def _domain_names(config: Mapping[str, Any]) -> list[str]:
    configured = config.get("benchmark", {}).get("domain_names", ())
    if len(configured) == 4:
        return [str(value) for value in configured]
    return [
        "clear_day_citystreet",
        "day_highway",
        "night",
        "rainy_day",
    ]




def _hyperparameter_lines(config: Mapping[str, Any]) -> list[str]:
    benchmark = config.get("benchmark", {})
    model = config.get("model", {})
    training = config.get("training", {})
    replay = config.get("replay_buffer", {})
    distillation = config.get("distillation", {})
    controller = config.get("controller", {})
    reward = config.get("reward", {})
    resources = config.get("resources", {})
    loading = config.get("data_loading", {})
    fixed_replay = "replay_loss_weight_levels" in training
    architecture = str(model.get("architecture", "fasterrcnn_resnet50_fpn_v2"))
    raw_weights = model.get("weights", "DEFAULT")
    random_initialization = raw_weights is None or str(raw_weights).upper() in {
        "NONE",
        "RANDOM",
        "FALSE",
    }
    initialization = (
        "random initialization"
        if random_initialization
        else "complete Torchvision COCO Faster R-CNN R50-FPN V2"
    )
    compile_description = (
        f"{training.get('compile_dynamic', True)} (ResNet body and FPN)"
    )
    optimizer_detail = (
        "ResNet stem/layer1-4 LR groups; heads/FPN/adapters at base LR"
        if training.get("stagewise_protection", False)
        else "N/A"
    )
    replay_capacity_lines = (
        [f"  Fixed Capacity: {replay.get('max_capacity', 250)} images"]
        if fixed_replay
        else [
            f"  Initial Capacity: {replay.get('initial_capacity', 'N/A')}",
            f"  Minimum Capacity: {replay.get('minimum_capacity', 'N/A')}",
            f"  Maximum Capacity: {replay.get('max_capacity', 'N/A')}",
        ]
    )
    lines = [
        "HYPERPARAMETERS",
        SEPARATOR,
        f"Seed: {config.get('seed', 0)}",
        f"Benchmark: {benchmark.get('name', 'CLAD-D')}",
        f"Data Root: {benchmark.get('data_root', 'N/A')}",
        f"CLAD Repo Root: {benchmark.get('clad_repo_root', 'N/A')}",
        f"Number of Domains: {benchmark.get('num_tasks', 4)}",
        f"Foreground Classes: {benchmark.get('num_foreground_classes', 6)}",
        "",
        "Model:",
        f"  Architecture: {architecture}",
        f"  Weights: {model.get('weights', 'DEFAULT')}",
        f"  Initialization: {initialization}",
        f"  Input Resize: min={model.get('min_size', 'N/A')}, max={model.get('max_size', 'N/A')}",
        f"  Adapter Reduction: {model.get('adapter_reduction', 'N/A')}",
        f"  Adapter Merge: {model.get('adapter_merge', 'parallel')}",
        f"  Adapter Groups: {', '.join(model.get('adapters', {}).get('groups', ())) or 'none'}",
        f"  Adapter Generation Limits: {model.get('adapters', {}).get('max_generations', 'N/A')}",
        f"  Class-Balanced RoI Loss: {model.get('class_balanced_roi_loss', True)}",
        f"  Preserve Exact COCO Predictor Rows: {model.get('preserve_coco_predictor_rows', True)}",
        "",
        "Training:",
        f"  Batch Size: {training.get('batch_size', 'N/A')}",
        f"  Evaluation Batch Size: {training.get('eval_batch_size', 'N/A')}",
        f"  Epochs per Domain: {training.get('epochs_per_task', 'N/A')}",
        f"  Optimizer: {training.get('optimizer', 'N/A')}",
        f"  Learning Rate: {training.get('learning_rate', 'N/A')}",
        f"  Momentum: {training.get('momentum', 'N/A')}",
        f"  Weight Decay: {training.get('weight_decay', 'N/A')}",
        f"  Optimizer Parameter Groups: {optimizer_detail}",
        f"  Scheduler: {training.get('scheduler', 'N/A')}",
        f"  Scheduler Milestones: {training.get('scheduler_milestones', 'N/A')}",
        f"  Warmup Epochs: {training.get('warmup_epochs', 0)}",
        f"  Initial Replay Loss Weight: {training.get('replay_loss_weight', 1.0)}",
        f"  Replay Loss Weight Levels: {training.get('replay_loss_weight_levels', 'N/A')}",
        f"  Class-Balanced Current-Image Sampling: {training.get('rare_image_sampling', True)}",
        f"  Current-Image Balance Strength: {training.get('rare_image_sampling_strength', 'N/A')}",
        f"  Class-Balance Beta / Cap: {training.get('class_balance_beta', 'N/A')} / {training.get('class_balance_max_weight', 'N/A')}",
        f"  Stagewise Backbone Protection after Task 1: {training.get('stagewise_protection', False)}",
        f"  Backbone Stage LR Multipliers after Task 1: {training.get('resnet_stage_lr_multipliers_after_task1', 'N/A')}",
        "  AMP: True",
        f"  Compilation: {compile_description}",
        f"  Decision Frequency: every {training.get('decision_frequency_epochs', 'N/A')} epochs",
        "",
        "Replay Distillation:",
        f"  Enabled: {distillation.get('enabled', False)}",
        f"  Scope: {distillation.get('scope', 'N/A')}",
        f"  Loss Weight / Temperature: {distillation.get('loss_weight', 'N/A')} / {distillation.get('temperature', 'N/A')}",
        f"  Cached Proposals per Image: {distillation.get('max_proposals_per_image', 'N/A')}",
        f"  Proposal Match IoU: {distillation.get('match_iou_threshold', 'N/A')}",
        f"  Cache Dtype: {distillation.get('cache_dtype', 'N/A')}",
        "",
        "Replay Buffer:",
        f"  Enabled: {replay.get('enabled', True)}",
        *replay_capacity_lines,
        f"  Replay Batch Size: {replay.get('replay_batch_size', 'N/A')}",
        f"  Retained-Image Refinement Candidates: {replay.get('refinement_candidate_count', 'N/A')}",
        f"  Utility Sampling Strength: {replay.get('utility_sampling_strength', 0.0)}",
        f"  Domain/Class-Balanced Sampling: {replay.get('balanced_sampling', True)}",
        f"  Hierarchical Domain-First/Class-Balanced Admission: {replay.get('class_balanced_admission', True)}",
        f"  Batch-Aware Replay Sampling: {replay.get('batch_aware_sampling', True)}",
        f"  Balance Strength / Cap: {replay.get('balanced_sampling_strength', 'N/A')} / {replay.get('balanced_sampling_max_weight', 'N/A')}",
        (
            "  Replay Semantics: completed earlier domains only"
            if fixed_replay
            else "  Replay Semantics: legacy online admission"
        ),
        "",
        "DQN Controller:",
        (
            "  Feedback: support-aware AP, weight n/(n+"
            f"{controller['feedback_support_scale']:g}); official AP unchanged"
            if "feedback_support_scale" in controller
            else "  Feedback: legacy unweighted AP"
        ),
        f"  Batch Size: {controller.get('batch_size', 'N/A')}",
        f"  Learning Starts: {controller.get('learning_starts', 'N/A')}",
        f"  Gradient Steps per Decision: {controller.get('gradient_steps', 'N/A')}",
        f"  Hidden Dimensions: {controller.get('hidden_dims', 'N/A')}",
        f"  Epsilon: {controller.get('epsilon_start', 'N/A')} -> {controller.get('epsilon_end', 'N/A')}",
        f"  Exploration Steps: {controller.get('exploration_steps', 'N/A')}",
        f"  Structural Expansion Cooldown: {controller.get('structural_expansion_cooldown_decisions', 'N/A')} decisions",
        f"  Post-Freeze Cooldown: {controller.get('freeze_cooldown_epochs', 0)} training epochs",
        f"  Replay-Weight Freeze: {controller.get('replay_weight_freeze_epochs', 0)} training epochs",
        (
            f"  AP Feedback EMA Alpha: {controller.get('feedback_ema_alpha', 0.3)}"
            if fixed_replay
            else "  AP Feedback EMA Alpha: N/A (legacy controller)"
        ),
        "",
        "Reward Weights:",
        f"  Current / Old / Rare AP Gain: {reward.get('current_ap_gain', 'N/A')} / {reward.get('old_ap_gain', 'N/A')} / {reward.get('rare_ap_gain', 'N/A')}",
        f"  Forgetting / Parameter Delta: {reward.get('forgetting_delta', 'N/A')} / {reward.get('parameter_delta', 'N/A')}",
        f"  Invalid Action: {reward.get('invalid_action', 'N/A')}",
        "",
        "Resources and Data Loading:",
        f"  Maximum Parameter Ratio: {resources.get('max_params_ratio', 'N/A')}",
        f"  Data Workers: {loading.get('num_workers', 'N/A')}",
        f"  Pin Memory: {loading.get('pin_memory', 'N/A')}",
        f"  Prefetch Factor: {loading.get('prefetch_factor', 'N/A')}",
        "",
    ]


    if controller.get("algorithm", "dqn") == "fixed_schedule":
        start = lines.index("DQN Controller:")
        end = lines.index("Resources and Data Loading:")
        lines[start:end] = [
            "CARL-FS Fixed Schedule (no RL):",
            "  Expansion: one eligible generation per group at Tasks 2-4 start",
            "  Group order: backbone -> fpn -> roi; generation/parameter caps enforced",
            "  Structural cooldown: not applied to scheduled allocation",
            "  Temporary freezing: disabled; stagewise protection unchanged",
            f"  Fixed replay weight: {training.get('replay_loss_weight', 1.5)}",
            "  Refinement: every 2 epochs with historical replay; final boundary skipped",
            "  DQN selection/updates: disabled",
            "",
        ]
    return lines


def render_training_report(payload: Mapping[str, Any], path: Path) -> str:
    config = _config(payload)
    fixed_replay = "replay_loss_weight_levels" in config.get("training", {})
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise ValueError("Training report payload does not contain metrics")
    summary = metrics.get("controller_probe", {})
    test_metrics = payload.get("test_metrics", {})
    if not isinstance(test_metrics, Mapping):
        test_metrics = {}
    task_reports = list(metrics.get("task_reports", ()))
    class_names = _class_names(config)
    domain_names = _domain_names(config)

    lines = [
        SEPARATOR,
        "CARL-FS TRAINING RESULTS" if config.get("controller", {}).get("algorithm") == "fixed_schedule" else "CARL-D TRAINING RESULTS",
        f"Timestamp: {_timestamp_for(path)}",
        f"Checkpoint: {payload.get('checkpoint', 'N/A')}",
        "Metric: controller-probe CLAD-D 2023 COCO 101-point bounding-box "
        "AP at IoU 0.50",
        "Final Test CLAD-D 2023 Equal-Domain mAP50: "
        f"{_metric(test_metrics.get('equal_domain_map50'))}",
        "Final Controller-Probe CLAD-D 2023 Equal-Domain mAP50: "
        f"{_metric(summary.get('equal_domain_map50'))}",
        f"Anytime Equal-Domain mAP50: {_metric(summary.get('anytime_equal_domain_map50'))}",
        f"Forgetting: {_metric(summary.get('forgetting'))}",
        f"Backward Transfer: {_metric(summary.get('backward_transfer'))}",
        f"Plasticity: {_metric(summary.get('plasticity'))}",
        (
            "Training Time: "
            f"{float(payload['training_time_seconds']) / 3600.0:.2f} hours "
            f"({float(payload['training_time_seconds']):.0f} seconds)"
            if payload.get("training_time_seconds") is not None
            else "Training Time: N/A (not recorded in checkpoint)"
        ),
        SEPARATOR,
        "",
    ]
    lines.extend(_hyperparameter_lines(config))

    lines.extend(["CONTROLLER-PROBE AP50 MATRIX", SEPARATOR])
    matrix = summary.get("ap50_matrix", ())
    for task_id, row in enumerate(matrix):
        values = [
            f"D{domain_id + 1}={_metric(value)}" for domain_id, value in enumerate(row)
        ]
        lines.append(f"After Task {task_id + 1}: " + "  ".join(values))
    lines.append("")

    lines.extend(["PER-TASK CONTROLLER-PROBE PERFORMANCE", SEPARATOR])
    for report in task_reports:
        task_id = int(report.get("task_id", len(lines)))
        probe = report.get("controller_probe", {})
        lines.append(f"Task {task_id + 1}:")
        lines.append(
            f"  Equal-Domain mAP50: {_metric(probe.get('equal_domain_map50'))}"
        )
        per_domain = probe.get("per_domain", {})
        feedback = report.get("controller_feedback", {})
        if feedback.get("feedback_convention") in {
            "support_aware_v1", "support_aware_v2"
        }:
            lines.append(
                "  RL Support-Aware AP (not benchmark mAP): "
                f"{_metric(feedback.get('seen_domain_ap50'))}"
            )
            lines.append(
                "  RL Support-Aware Rare AP / Forgetting: "
                f"{_metric(feedback.get('rare_class_ap50'))} / "
                f"{_metric(feedback.get('forgetting'))}"
            )
            if feedback.get("feedback_convention") == "support_aware_v2":
                lines.append(
                    "  RL Rare AP Aggregation: equal mean of support-weighted class APs"
                )
            else:
                lines.append(
                    "  RL Rare AP Aggregation: pooled support-weighted class/domain cells"
                )
        for domain_id in range(min(task_id + 1, len(domain_names))):
            domain = _value(per_domain, domain_id, {})
            lines.append(
                f"  Domain {domain_id + 1} ({domain_names[domain_id]}): "
                f"{_metric(domain.get('map50'))}"
            )
        if "replay_images_available_during_task" in report:
            during_counts = report.get("replay_domain_counts_during_task", {})
            after_counts = report.get("replay_domain_counts_after_admission", {})

            def render_counts(counts: Mapping[Any, Any]) -> str:
                rendered = ", ".join(
                    f"D{int(domain) + 1}={int(count)}"
                    for domain, count in sorted(
                        ((int(key), value) for key, value in counts.items())
                    )
                )
                return rendered or "none"

            lines.append(
                "  Replay Available During Task: "
                f"{report.get('replay_images_available_during_task', 'N/A')}/"
                f"{report.get('replay_capacity', 'N/A')} images "
                f"({render_counts(during_counts)})"
            )
            lines.append(
                "  Replay After Task Admission: "
                f"{report.get('replay_images_after_admission', 'N/A')}/"
                f"{report.get('replay_capacity', 'N/A')} images "
                f"({render_counts(after_counts)})"
            )
            lines.append(
                "  Completed-Task Candidates Considered / Retained: "
                f"{report.get('admitted_candidates_from_task', 'N/A')} / "
                f"{report.get('retained_from_task', 'N/A')}"
            )
            class_image_counts = report.get(
                "replay_class_image_counts_after_admission", {}
            )
            class_object_counts = report.get(
                "replay_class_object_counts_after_admission", {}
            )
            object_size_counts = report.get(
                "replay_object_size_counts_after_admission", {}
            )
            lines.append(
                "  Replay Class Image Counts: "
                + ", ".join(
                    f"C{int(class_id)}={int(count)}"
                    for class_id, count in sorted(
                        (int(key), value) for key, value in class_image_counts.items()
                    )
                )
            )
            lines.append(
                "  Replay Class Object Counts: "
                + ", ".join(
                    f"C{int(class_id)}={int(count)}"
                    for class_id, count in sorted(
                        (int(key), value) for key, value in class_object_counts.items()
                    )
                )
            )
            lines.append(
                "  Replay Object Size Counts: "
                + ", ".join(
                    f"{name}={int(count)}" for name, count in object_size_counts.items()
                )
            )
            lines.append(
                "  Replay Loss Weight: "
                f"{report.get('replay_loss_weight_start', 'N/A')} -> "
                f"{report.get('replay_loss_weight_end', 'N/A')}"
            )
            lines.append(
                "  Cached Teacher Targets for Next Task: "
                f"{report.get('distillation_teacher_targets', 'N/A')}"
            )
            lines.append(
                "  Mean Replay KD Loss / Matched Proposals per Batch: "
                f"{_metric(report.get('mean_replay_distillation_loss'))} / "
                f"{_metric(report.get('mean_distillation_matches_per_batch'))}"
            )
        else:
            lines.append(
                f"  Replay: {report.get('replay_images', 'N/A')}/"
                f"{report.get('replay_capacity', 'N/A')} images"
            )
        lines.append(f"  Model Parameters: {int(report.get('model_parameters', 0)):,}")
    lines.append("")

    lines.extend(["PER-CLASS EQUAL-DOMAIN AP50 AT END OF EACH TASK", SEPARATOR])
    for report in task_reports:
        task_id = int(report.get("task_id", 0))
        per_class = report.get("controller_probe", {}).get(
            "per_class_equal_domain_ap50", {}
        )
        lines.append(f"Task {task_id + 1}:")
        for class_id, name in class_names.items():
            lines.append(f"  {name}: {_metric(_value(per_class, class_id))}")
    lines.append("")

    final_probe = task_reports[-1].get("controller_probe", {}) if task_reports else {}
    lines.extend(["FINAL PER-DOMAIN PER-CLASS AP50", SEPARATOR])
    final_domains = final_probe.get("per_domain", {})
    for domain_id, domain_name in enumerate(domain_names):
        domain = _value(final_domains, domain_id, {})
        if not domain:
            continue
        lines.append(
            f"Domain {domain_id + 1} ({domain_name}) - "
            f"mAP50: {_metric(domain.get('map50'))}, "
            f"Images: {domain.get('num_images', 'N/A')}"
        )
        per_class = domain.get("per_class_ap50", {})
        counts = domain.get("target_count_per_class", {})
        for class_id, name in class_names.items():
            lines.append(
                f"  {name}: {_metric(_value(per_class, class_id))} "
                f"(objects={_value(counts, class_id, 0)})"
            )
    lines.append("")

    lines.extend(["FINAL PER-CLASS EQUAL-DOMAIN AP50", SEPARATOR])
    final_per_class = final_probe.get("per_class_equal_domain_ap50", {})
    for class_id, name in class_names.items():
        lines.append(f"{name}: {_metric(_value(final_per_class, class_id))}")
    lines.append("")

    if test_metrics:
        lines.extend(
            [
                "FINAL TEST PERFORMANCE",
                SEPARATOR,
                "CLAD-D 2023 Equal-Domain mAP50: "
                f"{_metric(test_metrics.get('equal_domain_map50'))}",
                (
                    "Test Evaluation Time: "
                    f"{float(payload['test_evaluation_time_seconds']):.0f} seconds"
                    if payload.get("test_evaluation_time_seconds") is not None
                    else "Test Evaluation Time: N/A"
                ),
                "",
                "Per-Class Equal-Domain AP50:",
            ]
        )
        test_per_class = test_metrics.get("per_class_equal_domain_ap50", {})
        for class_id, name in class_names.items():
            lines.append(f"  {name}: {_metric(_value(test_per_class, class_id))}")
        lines.append("")
        lines.append("Per-Domain Per-Class AP50:")
        test_domains = test_metrics.get("per_domain", {})
        for domain_id, domain_name in enumerate(domain_names):
            domain = _value(test_domains, domain_id, {})
            if not domain:
                continue
            lines.append(
                f"  Domain {domain_id + 1} ({domain_name}) - "
                f"mAP50: {_metric(domain.get('map50'))}, "
                f"Images: {domain.get('num_images', 'N/A')}"
            )
            per_class = domain.get("per_class_ap50", {})
            counts = domain.get("target_count_per_class", {})
            for class_id, name in class_names.items():
                lines.append(
                    f"    {name}: {_metric(_value(per_class, class_id))} "
                    f"(objects={_value(counts, class_id, 0)})"
                )
        lines.append("")

    base = int(metrics.get("base_model_parameters", 0))
    final = int(metrics.get("model_parameters", 0))
    maximum = int(metrics.get("max_model_parameters", 0))
    if test_metrics:
        _append_environment_results(lines, test_metrics, class_names)
    lines.extend(
        [
            "MODEL AND REPLAY STATISTICS",
            SEPARATOR,
            f"Optimizer Steps: {int(metrics.get('optimizer_steps', 0)):,}",
            f"Base Model Parameters: {base:,}",
            f"Final Model Parameters: {final:,}",
            f"Maximum Model Parameters: {maximum:,}",
            f"Parameter Growth: {final - base:,}",
            f"Final/Base Ratio: {(final / base if base else 0.0):.4f}",
            (
                "Final Post-Admission Replay Images: "
                if fixed_replay
                else "Replay Images: "
            )
            + str(metrics.get("replay_images", "N/A")),
            ("Fixed Replay Capacity: " if fixed_replay else "Replay Capacity: ")
            + str(metrics.get("replay_capacity", "N/A")),
            "Final Replay Class Image Counts: "
            + str(metrics.get("replay_class_image_counts", {})),
            "Final Replay Class Object Counts: "
            + str(metrics.get("replay_class_object_counts", {})),
            "Final Replay Object Size Counts: "
            + str(metrics.get("replay_object_size_counts", {})),
            f"Final Replay Loss Weight: {metrics.get('replay_loss_weight', 'N/A')}",
            f"Allowed Replay Loss Weights: {metrics.get('replay_loss_weight_levels', 'N/A')}",
            "",
            "FIXED-SCHEDULE ACTION SUMMARY" if config.get("controller", {}).get("algorithm") == "fixed_schedule" else "DQN ACTION SUMMARY",
            SEPARATOR,
        ]
    )
    action_distribution = metrics.get("action_distribution", {})
    for action_name in sorted(action_distribution):
        lines.append(f"{action_name}: {int(action_distribution[action_name])}")
    actions = list(metrics.get("actions", ()))
    lines.append(f"Total Action Executions: {len(actions)}")
    lines.append(
        f"Invalid Decisions: {sum(bool(item.get('invalid')) for item in actions)}"
    )
    changed_by_action: dict[str, int] = Counter(
        str(item.get("action_name", "unknown"))
        for item in actions
        if bool(item.get("changed"))
    )
    lines.append("")
    is_fs = config.get("controller", {}).get("algorithm") == "fixed_schedule"
    lines.append("Changed Executions:" if is_fs else "Changed Executions and Mean Observed Reward:")
    mean_rewards = metrics.get("mean_reward_by_action", {})
    for action_name in sorted(action_distribution):
        mean_reward = _value(mean_rewards, action_name)
        lines.append(
            f"  {action_name}: changed={changed_by_action.get(action_name, 0)}"
            + ("" if is_fs else f", mean_reward={_metric(mean_reward)}")
        )

    diagnostics = metrics.get("controller_action_diagnostics", {})
    decisions = (
        list(diagnostics.get("decisions", ()))
        if isinstance(diagnostics, Mapping)
        else []
    )
    selection_modes = Counter(
        str(item.get("selection_mode", "unknown")) for item in decisions
    )
    if selection_modes:
        lines.append("")
        lines.append(
            "Selection Modes: "
            + ", ".join(
                f"{name}={count}" for name, count in sorted(selection_modes.items())
            )
        )

    by_task: dict[int, Counter[str]] = defaultdict(Counter)
    for event in actions:
        by_task[int(event.get("applies_to_task_id", event.get("task_id", 0)))][
            str(event.get("action_name", "unknown"))
        ] += 1
    lines.append("")
    lines.append("Actions Per Task:")
    for task_id in range(int(metrics.get("completed_tasks", len(task_reports)))):
        rendered = ", ".join(
            f"{name}={count}" for name, count in sorted(by_task[task_id].items())
        )
        lines.append(f"  Task {task_id + 1}: {rendered or 'none'}")

    lines.extend(["", SEPARATOR, ""])
    return "\n".join(lines)


def write_training_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_training_report(payload, output), encoding="utf-8")
    return output


def _append_environment_results(lines, metrics, class_names):
    environment = metrics.get("per_environment", {})
    if not environment:
        return
    names = ("Day", "Night", "Highway", "City", "Country")
    lines.extend([
        "FINAL TEST ENVIRONMENTAL AP50", SEPARATOR,
        "COCO 101-point interpolation at IoU 0.50; values in [0, 1].",
        "Overall is the four-domain equal mean, not the mean of these overlapping groups.",
        "City = Citystreet; Country = Countryroad.",
        "Model | Overall | " + " | ".join(names),
        str(metrics.get("model_name", "CARL-D")) + " | " + _metric(metrics.get("equal_domain_map50")) + " | "
        + " | ".join(_metric(environment.get(name, {}).get("map50")) for name in names),
        "",
    ])
    for name in names:
        group = environment.get(name, {})
        lines.append(
            f"{name} - mAP50: {_metric(group.get('map50'))}, "
            f"Images: {group.get('num_images', 'N/A')}"
        )
        for class_id, class_name in class_names.items():
            lines.append(
                f"  {class_name}: "
                f"{_metric(_value(group.get('per_class_ap50', {}), class_id))} "
                f"(objects={_value(group.get('target_count_per_class', {}), class_id, 0)})"
            )
    lines.append("")


def render_test_report(payload: Mapping[str, Any], path: Path) -> str:
    config = _config(payload)
    metrics = payload.get("metrics", {})
    class_names = _class_names(config)
    domain_names = _domain_names(config)
    lines = [
        SEPARATOR,
        "CARL-FS FINAL TEST RESULTS" if config.get("controller", {}).get("algorithm") == "fixed_schedule" else "CARL-D FINAL TEST RESULTS",
        f"Timestamp: {_timestamp_for(path)}",
        f"Checkpoint: {payload.get('checkpoint', 'N/A')}",
        f"Metric: {metrics.get('metric', 'COCO-style bounding-box AP at IoU 0.50')}",
        f"CLAD-D 2023 Equal-Domain mAP50: {_metric(metrics.get('equal_domain_map50'))}",
        SEPARATOR,
        "",
        "PER-DOMAIN PER-CLASS COCO 101-POINT AP50",
        SEPARATOR,
    ]
    per_domain = metrics.get("per_domain", {})
    for domain_id, domain_name in enumerate(domain_names):
        domain = _value(per_domain, domain_id, {})
        if not domain:
            continue
        lines.append(
            f"Domain {domain_id + 1} ({domain_name}) - "
            f"mAP50: {_metric(domain.get('map50'))}, "
            f"Images: {domain.get('num_images', 'N/A')}"
        )
        per_class = domain.get("per_class_ap50", {})
        counts = domain.get("target_count_per_class", {})
        for class_id, name in class_names.items():
            lines.append(
                f"  {name}: {_metric(_value(per_class, class_id))} "
                f"(objects={_value(counts, class_id, 0)})"
            )
    lines.extend(["", "PER-CLASS EQUAL-DOMAIN COCO 101-POINT AP50", SEPARATOR])
    per_class = metrics.get("per_class_equal_domain_ap50", {})
    for class_id, name in class_names.items():
        lines.append(f"{name}: {_metric(_value(per_class, class_id))}")
    lines.append("")
    lines.extend([SEPARATOR, ""])
    _append_environment_results(lines, metrics, class_names)
    return "\n".join(lines)


def write_test_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_test_report(payload, output), encoding="utf-8")
    return output


__all__ = [
    "render_test_report",
    "render_training_report",
    "write_test_report",
    "write_training_report",
]
