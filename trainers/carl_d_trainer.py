"""Sequential, detection-only CARL-D training on the four CLAD-D domains."""

from __future__ import annotations

import math
import os
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import torch
from PIL import Image
from torchvision.ops import box_iou
from tqdm.auto import tqdm

from agents import (
    ACTION_NAMES,
    EXPANSION_ACTION_GROUPS,
    CARLDAction,
    CARLDDQNController,
    create_dqn_controller,
)
from benchmarks import CLADDBenchmark, create_cladd_benchmark
from agents.fixed_schedule import FixedScheduleController
from buffers import DetectionReplayBuffer, DetectionReplayRecord
from models import (
    RESNET50_ARCHITECTURE,
    SUPPORTED_ARCHITECTURES,
    CARLDDetector,
    create_carl_d_detector,
)
from utils.checkpointing import (
    CHECKPOINT_TYPE,
    CHECKPOINT_VERSION,
)
from utils.checkpointing import (
    restore_checkpoint as restore_carl_d_checkpoint,
)
from utils.checkpointing import (
    save_checkpoint as save_carl_d_checkpoint,
)
from utils.detection_metrics import (
    AP50Result,
    DomainAP50History,
    DomainAP50Summary,
    evaluate_domains_ap50,
    evaluate_environment_ap50,
)

from .config_parser import (
    FOREGROUND_CLASS_IDS,
    dataset_split_fingerprint,
    resume_config_fingerprint,
    validate_carl_d_config,
    normalize_carl_d_config,
)
from .controller_feedback import support_aware_ap
from .events import ActionExecutionEvent
from .runtime import (
    _CudaReplayTransfer,
    _box_iou_matrix,
    _cpu_detection_batch,
    _cpu_detection_dict,
    _finite_or_zero,
    _make_grad_scaler,
    _seed_everything,
    _to_device,
    check_model_health,
)


class CARLDTrainer:
    """Train a single shared detector sequentially over all CLAD-D domains.

    Native RPN/box-regression losses and a mildly class-balanced positive-RoI
    category loss are used without an image-level objective. Replay stores
    complete images with all boxes. A masked DQN acts
    only at explicit two-epoch boundaries. The complete four-domain stream is
    one RL episode so persistent replay and architecture actions can receive
    credit for their effects on later domains.
    """

    def __init__(
        self,
        *,
        model: CARLDDetector,
        benchmark: CLADDBenchmark,
        replay_buffer: DetectionReplayBuffer,
        controller: CARLDDQNController | FixedScheduleController,
        config: Mapping[str, Any],
        device: str | torch.device,
        domain_evaluator: Callable[..., DomainAP50Summary] = evaluate_domains_ap50,
    ) -> None:
        config = normalize_carl_d_config(config)
        validate_carl_d_config(config)
        self.config = config
        self.feedback_support_scale = float(
            self.config.get("controller", {}).get("feedback_support_scale", 20.0)
        )
        self.config.setdefault("controller", {})["feedback_support_scale"] = (
            self.feedback_support_scale
        )
        self._feedback_annotation_counts: Dict[int, Dict[int, int]] = {}
        controller_config = self.config.setdefault("controller", {})
        self.freeze_cooldown_epochs = int(
            controller_config.setdefault("freeze_cooldown_epochs", 2)
        )
        self.replay_weight_freeze_epochs = int(
            controller_config.setdefault("replay_weight_freeze_epochs", 4)
        )
        self.freeze_available_at_epoch = 0
        self.replay_weight_available_at_epoch = 0
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {self.device}")

        self.model = model.to(self.device)
        self.run_timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        self.benchmark = benchmark
        self.replay_buffer = replay_buffer
        self.controller = controller
        self.fixed_schedule = (
            self.config.get("controller", {}).get("algorithm", "dqn") == "fixed_schedule"
        )
        self.domain_evaluator = domain_evaluator

        self.foreground_class_ids = tuple(
            int(class_id)
            for class_id in self.config.get("evaluation", {}).get(
                "foreground_class_ids", FOREGROUND_CLASS_IDS
            )
        )
        if self.foreground_class_ids != FOREGROUND_CLASS_IDS:
            raise ValueError("CARL-D evaluation requires foreground IDs 1..6")

        if int(getattr(benchmark, "num_tasks", 4)) != 4:
            raise ValueError("CARL-D requires the four-domain CLAD-D stream")
        if int(getattr(model, "num_classes", 7)) != 7:
            raise ValueError("CARL-D requires background plus six foreground outputs")

        training = self.config.get("training", {})
        replay = self.config.get("replay_buffer", {})
        resources = self.config.get("resources", {})
        adapters = self.config.get("model", {}).get("adapters", {})
        self.compile_dynamic = training.get("compile_dynamic", True)
        self._replay_transfer_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self.cudnn_benchmark = bool(training.get("cudnn_benchmark", False))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = self.cudnn_benchmark
        compile_target = getattr(self.model, "backbone", None)
        body = getattr(compile_target, "body", None)
        fpn = getattr(compile_target, "fpn", None)
        immutable_modules = (body, fpn)
        if any(
            not callable(getattr(module, "compile", None))
            for module in immutable_modules
        ):
            raise RuntimeError(
                "CARL-D requires a compilable immutable detector backbone"
            )
        for module in immutable_modules:
            module.compile(dynamic=self.compile_dynamic)

        self.epochs_per_task = int(training.get("epochs_per_task", 50))
        self.warmup_epochs = int(training.get("warmup_epochs", 0))
        self.warmup_start_factor = float(training.get("warmup_start_factor", 1e-3))
        self._warmup_optimizer_step = 0
        self._warmup_total_optimizer_steps = 0
        self.replay_loss_weight = float(training.get("replay_loss_weight", 1.0))
        self.replay_loss_weight_levels = tuple(
            float(value)
            for value in training.get("replay_loss_weight_levels", (0.5, 1.0, 2.0))
        )
        self.class_balance_beta = float(training.get("class_balance_beta", 0.999))
        self.class_balance_max_weight = float(
            training.get("class_balance_max_weight", 3.0)
        )
        self.class_balanced_roi_loss = bool(
            self.config.get("model", {}).get("class_balanced_roi_loss", True)
        )
        self.stagewise_protection = bool(training.get("stagewise_protection", False))
        self.resnet_stage_lr_multipliers = tuple(
            float(v)
            for v in training.get(
                "resnet_stage_lr_multipliers_after_task1", (0.0, 0.0, 0.05, 0.25, 0.5)
            )
        )
        distillation = self.config.get("distillation", {})
        self.distillation_enabled = bool(distillation.get("enabled", False))
        self.distillation_loss_weight = float(distillation.get("loss_weight", 0.5))
        self.distillation_temperature = float(distillation.get("temperature", 2.0))
        self.distillation_max_proposals = int(
            distillation.get("max_proposals_per_image", 256)
        )
        self.distillation_match_iou = float(
            distillation.get("match_iou_threshold", 0.5)
        )
        self.distillation_cache_dtype = torch.float16
        self.decision_frequency_epochs = int(
            training.get("decision_frequency_epochs", 2)
        )
        if self.epochs_per_task <= 0:
            raise ValueError("epochs_per_task must be positive")
        if (
            len(self.replay_loss_weight_levels) < 2
            or tuple(sorted(set(self.replay_loss_weight_levels)))
            != self.replay_loss_weight_levels
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.replay_loss_weight_levels
            )
        ):
            raise ValueError(
                "replay_loss_weight_levels must contain at least two unique, "
                "increasing, finite positive values"
            )
        if not any(
            math.isclose(self.replay_loss_weight, value)
            for value in self.replay_loss_weight_levels
        ):
            raise ValueError("replay_loss_weight must be one configured level")
        if (
            self.decision_frequency_epochs <= 0
            or self.epochs_per_task % self.decision_frequency_epochs
        ):
            raise ValueError(
                "epochs_per_task must be a positive multiple of "
                "decision_frequency_epochs"
            )

        self.replay_enabled = bool(replay.get("enabled", True))
        self.replay_batch_size = int(replay.get("replay_batch_size", 2))
        if (
            self.replay_buffer.current_capacity != 250
            or self.replay_buffer.max_capacity != 250
        ):
            raise ValueError("primary CARL-D requires a fixed 250-image replay buffer")
        self.balanced_replay_sampling = bool(replay.get("balanced_sampling", True))
        self.class_balanced_replay_admission = bool(
            replay.get("class_balanced_admission", True)
        )
        self.batch_aware_replay_sampling = bool(
            replay.get("batch_aware_sampling", True)
        )
        self.balanced_replay_sampling_strength = float(
            replay.get("balanced_sampling_strength", 0.5)
        )
        self.balanced_replay_sampling_max_weight = float(
            replay.get("balanced_sampling_max_weight", 3.0)
        )
        self.refinement_candidate_count = int(
            replay.get("refinement_candidate_count", 50)
        )
        self.refinement_entropy_weight = float(
            replay.get("refinement_entropy_weight", 1.0)
        )
        self.refinement_localization_weight = float(
            replay.get("refinement_localization_weight", 1.0)
        )
        self.refinement_rare_class_weight = float(
            replay.get("refinement_rare_class_weight", 1.0)
        )
        self.refinement_feature_redundancy_weight = float(
            replay.get("refinement_feature_redundancy_weight", 0.1)
        )
        self.utility_sampling_strength = float(
            replay.get("utility_sampling_strength", 0.25)
        )
        self.decoded_image_cache_size = int(replay.get("decoded_image_cache_size", 320))
        self.decode_workers = int(replay.get("decode_workers", 4))
        if self.replay_batch_size <= 0:
            raise ValueError("replay batch size must be positive")
        if self.refinement_candidate_count < 0:
            raise ValueError("refinement_candidate_count must be nonnegative")
        if (
            not math.isfinite(self.refinement_entropy_weight)
            or not math.isfinite(self.refinement_localization_weight)
            or not math.isfinite(self.refinement_rare_class_weight)
            or not math.isfinite(self.refinement_feature_redundancy_weight)
            or self.refinement_entropy_weight < 0
            or self.refinement_localization_weight < 0
            or self.refinement_rare_class_weight < 0
            or self.refinement_feature_redundancy_weight < 0
        ):
            raise ValueError("replay refinement weights must be nonnegative")
        if (
            not math.isfinite(self.utility_sampling_strength)
            or not 0.0 <= self.utility_sampling_strength <= 1.0
        ):
            raise ValueError("utility_sampling_strength must be in [0, 1]")

        controller_config = self.config.get("controller", {})
        self.structural_expansion_cooldown_decisions = int(
            controller_config.get("structural_expansion_cooldown_decisions", 1)
        )

        self.adapters_enabled = bool(adapters.get("enabled", True))
        self.adapter_groups = tuple(
            str(group) for group in adapters.get("groups", ("backbone", "fpn", "roi"))
        )
        unknown_groups = set(self.adapter_groups) - {"backbone", "fpn", "roi"}
        if unknown_groups:
            raise ValueError(f"Unknown adapter groups: {sorted(unknown_groups)}")
        raw_generation_limits = adapters.get(
            "max_generations", {"backbone": 3, "fpn": 3, "roi": 4}
        )
        self.adapter_generation_limits = {
            str(group): int(limit) for group, limit in raw_generation_limits.items()
        }

        self.base_parameter_count = int(self.model.get_num_parameters())
        max_ratio = float(resources.get("max_params_ratio", 1.10))
        if not math.isfinite(max_ratio) or max_ratio < 1.0:
            raise ValueError("resources.max_params_ratio must be at least 1.0")
        self.max_parameter_count = int(
            math.floor(self.base_parameter_count * max_ratio)
        )
        self.parameter_growth_budget = max(
            1, self.max_parameter_count - self.base_parameter_count
        )

        # CUDA training always uses float16 autocast and gradient scaling.
        # AMP has no CUDA execution to accelerate on a CPU-only fallback.
        self.use_amp = self.device.type == "cuda"
        self.scaler = _make_grad_scaler(self.use_amp)
        self.current_task_id = -1
        self.optimizer = self._new_optimizer()
        self.scheduler = self._new_scheduler()

        self.metric_history = DomainAP50History(num_domains=4)
        self.evaluation_history: list[Dict[str, Any]] = []
        self.action_events: list[ActionExecutionEvent] = []
        self.task_reports: list[Dict[str, Any]] = []
        self.best_domain_ap50: Dict[int, float] = {}
        self.pending_decision: Optional[Dict[str, Any]] = None
        self.pending_outcome_metrics: Optional[Dict[str, Any]] = None
        self.pending_old_domain_results: Optional[Dict[int, Dict[str, Any]]] = None
        self.last_structural_expansion_decision_step: Optional[int] = None
        self.optimizer_step = 0
        self.next_task_id = 0
        self.current_epoch = 0
        self.task_optimizer_step = 0
        self.planned_task_optimizer_steps: Optional[int] = None
        self.at_domain_boundary = True
        self.loss_ema = 0.0
        self.recognition_loss_ema = 0.0
        self.localization_loss_ema = 0.0
        self.loss_ema_updates = 0
        self.latest_rpn_recall = 0.0
        self.dataset_fingerprint = dataset_split_fingerprint(self.benchmark)
        self.resume_config_fingerprint = resume_config_fingerprint(self.config)
        self._replay_source_cache: Dict[int, Dict[Any, Dict[str, Any]]] = {}
        self._refinement_cursor = 0
        self._task_class_count_cache: Dict[int, Dict[int, int]] = {}
        self._validation_loader_cache: Dict[int, Iterable[Any]] = {}
        self._decoded_image_cache: OrderedDict[str, Image.Image] = OrderedDict()
        self._decoded_image_cache_lock = threading.Lock()
        self._replay_tensor_cache: Dict[str, torch.Tensor] = {}
        self._replay_tensor_cache_lock = threading.Lock()
        self._distillation_target_cache: Dict[
            tuple[int, Any], Dict[str, torch.Tensor]
        ] = {}
        self._flipped_teacher_boxes: Dict[tuple[int, Any], torch.Tensor] = {}
        self._decode_executor = (
            ThreadPoolExecutor(
                max_workers=self.decode_workers,
                thread_name_prefix="carl-d-replay-decode",
            )
            if self.decode_workers > 0
            else None
        )

    def _progress(self, iterable: Iterable[Any], **kwargs: Any):
        """Create the always-enabled CARL-D progress bar."""
        kwargs.setdefault("mininterval", 0.5)
        return tqdm(
            iterable,
            disable=False,
            dynamic_ncols=True,
            **kwargs,
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        device: str | torch.device | None = None,
        *,
        load_pretrained_weights: bool = True,
    ) -> "CARLDTrainer":
        config = normalize_carl_d_config(config)
        validate_carl_d_config(config)
        if config.get("benchmark", {}).get("name") != "CLAD-D":
            raise ValueError("CARLDTrainer accepts only a CLAD-D configuration")
        seed = int(config.get("seed", 0))
        _seed_everything(seed)
        selected_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        benchmark = create_cladd_benchmark(config)
        model_config = dict(config.get("model", {}))
        architecture = str(
            model_config.get("architecture", RESNET50_ARCHITECTURE)
        ).lower()
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"CARL-D supports only {SUPPORTED_ARCHITECTURES}")
        model_config["architecture"] = architecture
        model_config.pop("adapters", None)
        if not load_pretrained_weights:
            model_config["weights"] = None
        allowed_model_keys = {
            "architecture",
            "weights",
            "num_classes",
            "adapter_reduction",
            "preserve_coco_predictor_rows",
            "adapter_merge",
            "class_balanced_roi_loss",
            "min_size",
            "max_size",
            "trainable_backbone_layers",
            "box_score_thresh",
            "box_nms_thresh",
            "box_detections_per_img",
        }
        unknown_model_keys = set(model_config) - allowed_model_keys
        if unknown_model_keys:
            raise ValueError(
                f"Unknown model configuration: {sorted(unknown_model_keys)}"
            )
        model = create_carl_d_detector(**model_config)

        replay_config = config.get("replay_buffer", {})
        replay_buffer = DetectionReplayBuffer(
            initial_capacity=int(replay_config.get("initial_capacity", 250)),
            max_capacity=int(replay_config.get("max_capacity", 250)),
            seed=seed + 10,
            foreground_class_ids=tuple(
                config.get("evaluation", {}).get(
                    "foreground_class_ids", FOREGROUND_CLASS_IDS
                )
            ),
        )
        controller = (
            FixedScheduleController()
            if config.get("controller", {}).get("algorithm", "dqn") == "fixed_schedule"
            else create_dqn_controller(config, str(selected_device))
        )
        return cls(
            model=model,
            benchmark=benchmark,
            replay_buffer=replay_buffer,
            controller=controller,
            config=config,
            device=selected_device,
        )

    def _backbone_stage_from_parameter_name(self, parameter_name: str) -> Optional[int]:
        marker = ".backbone.body."
        if marker not in parameter_name:
            return None  # Adapter banks stay plastic and use the base learning rate.
        name = parameter_name.split(marker, 1)[1].split(".", 1)[0]
        if name in {"conv1", "bn1"}:
            return 0
        if name in {"layer1", "layer2", "layer3", "layer4"}:
            return int(name[-1])
        return None

    def _new_optimizer(self) -> torch.optim.Optimizer:
        training = self.config.get("training", {})
        parameters = list(self.model.parameters())
        if self.stagewise_protection:
            # Keep optimizer group topology fixed across adapter expansion.
            grouped = {stage: [] for stage in (0, 1, 2, 3, 4, None)}
            for parameter_name, parameter in self.model.named_parameters():
                grouped[
                    self._backbone_stage_from_parameter_name(parameter_name)
                ].append(parameter)
            parameters = [
                {
                    "params": values,
                    "carl_backbone_stage": stage,
                    "carl_lr_scale": 1.0,
                }
                for stage, values in grouped.items()
            ]
        optimizer = torch.optim.SGD(
            parameters,
            lr=float(training.get("learning_rate", 0.005)),
            momentum=float(training.get("momentum", 0.9)),
            weight_decay=float(training.get("weight_decay", 5e-4)),
        )
        optimizer.register_step_post_hook(self._record_optimizer_step)
        return optimizer

    def _new_scheduler(self):
        training = self.config.get("training", {})
        return torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(v) for v in training.get("scheduler_milestones", (34, 44))],
            gamma=float(training.get("scheduler_gamma", 0.1)),
        )

    def _set_optimizer_lrs(self, learning_rates: Sequence[float]) -> None:
        rates = [float(value) for value in learning_rates]
        if len(rates) != len(self.optimizer.param_groups):
            raise ValueError("Learning-rate count does not match optimizer groups")
        for parameter_group, learning_rate in zip(self.optimizer.param_groups, rates):
            parameter_group["lr"] = learning_rate
        # Keep get_last_lr() and serialized scheduler metadata consistent with
        # the per-update warmup, which intentionally runs outside epoch steps.
        if hasattr(self.scheduler, "_last_lr"):
            self.scheduler._last_lr = list(rates)

    def _configure_stagewise_protection(self, task_id: int) -> None:
        if not self.stagewise_protection:
            for parameter_group in self.optimizer.param_groups:
                parameter_group["carl_lr_scale"] = 1.0
            return
        protected_task = self.stagewise_protection and int(task_id) > 0
        for parameter_name, parameter in self.model.named_parameters():
            stage = self._backbone_stage_from_parameter_name(parameter_name)
            if stage is None:
                continue
            multiplier = (
                self.resnet_stage_lr_multipliers[stage] if protected_task else 1.0
            )
            parameter.requires_grad = multiplier > 0.0
        for parameter_group in self.optimizer.param_groups:
            stage = parameter_group.get("carl_backbone_stage")
            parameter_group["carl_lr_scale"] = (
                self.resnet_stage_lr_multipliers[int(stage)]
                if protected_task and stage is not None
                else 1.0
            )

    def _enforce_stagewise_module_modes(self) -> None:
        if not self.stagewise_protection or self.current_task_id <= 0:
            return
        body = self.model.backbone.body
        names = (("conv1", "bn1"), ("layer1",), ("layer2",), ("layer3",), ("layer4",))
        for multiplier, stage_names in zip(self.resnet_stage_lr_multipliers, names):
            if multiplier == 0:
                for name in stage_names:
                    getattr(body, name).eval()
        return

    def _display_learning_rate(self) -> float:
        return max(
            (float(group["lr"]) for group in self.optimizer.param_groups),
            default=0.0,
        )

    def _reset_schedule_for_task(self, *, updates_per_epoch: int = 0) -> None:
        base_lr = float(self.config.get("training", {}).get("learning_rate", 0.005))
        for parameter_group in self.optimizer.param_groups:
            group_lr = base_lr * float(parameter_group.get("carl_lr_scale", 1.0))
            parameter_group["lr"] = group_lr
            parameter_group["initial_lr"] = group_lr
        self.scheduler = self._new_scheduler()
        self._warmup_optimizer_step = 0
        self._warmup_total_optimizer_steps = self.warmup_epochs * max(
            0, int(updates_per_epoch)
        )
        if self._warmup_total_optimizer_steps:
            self._set_optimizer_lrs(
                [base * self.warmup_start_factor for base in self.scheduler.base_lrs]
            )

    def _advance_warmup(self) -> None:
        if self._warmup_optimizer_step >= self._warmup_total_optimizer_steps:
            return
        self._warmup_optimizer_step += 1
        progress = self._warmup_optimizer_step / self._warmup_total_optimizer_steps
        factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
        self._set_optimizer_lrs([base * factor for base in self.scheduler.base_lrs])

    def _refresh_optimizer_after_expansion(self) -> None:
        previous_optimizer = self.optimizer
        previous_state = dict(previous_optimizer.state)
        previous_lrs = [group["lr"] for group in previous_optimizer.param_groups]
        scheduler_state = self.scheduler.state_dict()

        self.optimizer = self._new_optimizer()
        self._configure_stagewise_protection(self.current_task_id)
        self.scheduler = self._new_scheduler()
        for parameter_group in self.optimizer.param_groups:
            for parameter in parameter_group["params"]:
                if parameter in previous_state:
                    self.optimizer.state[parameter] = previous_state[parameter]
        # Expansion must not silently restart an LR trajectory. All supported
        # schedulers retain the same optimizer-group topology here, so an
        # incompatible state is a hard error.
        self.scheduler.load_state_dict(scheduler_state)
        if previous_lrs:
            self._set_optimizer_lrs(
                [
                    previous_lrs[min(index, len(previous_lrs) - 1)]
                    for index in range(len(self.optimizer.param_groups))
                ]
            )

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    def _detector_losses(
        self,
        images: Sequence[torch.Tensor],
        targets: Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device_images = [image.to(self.device, non_blocking=True) for image in images]
        device_targets = [_to_device(dict(target), self.device) for target in targets]
        with self._autocast():
            loss_dict = self.model(device_images, device_targets)
            if not isinstance(loss_dict, Mapping) or not loss_dict:
                raise RuntimeError("Detector training must return native loss terms")
            total = sum(loss for loss in loss_dict.values())
        return total, dict(loss_dict)

    def _replay_losses_with_distillation(
        self,
        replay_batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        device_images = [
            image.to(self.device, non_blocking=True) for image in replay_batch["images"]
        ]
        device_targets = [
            _to_device(dict(target), self.device) for target in replay_batch["targets"]
        ]
        teacher_targets = [
            {
                name: value.to(self.device, non_blocking=True)
                for name, value in target.items()
            }
            for target in self._teacher_targets_for_replay_batch(replay_batch)
        ]
        with self._autocast():
            loss_dict, distillation_loss, matched = (
                self.model.forward_with_replay_distillation(
                    device_images,
                    device_targets,
                    teacher_targets,
                    temperature=self.distillation_temperature,
                    match_iou_threshold=self.distillation_match_iou,
                )
            )
            if not isinstance(loss_dict, Mapping) or not loss_dict:
                raise RuntimeError("Replay detector must return native loss terms")
            detection_loss = sum(loss for loss in loss_dict.values())
        return detection_loss, dict(loss_dict), distillation_loss, matched

    @staticmethod
    def _decode_replay_image(path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB").copy()

    def _load_replay_image(self, path: str) -> Image.Image:
        key = os.path.abspath(path)
        if self.decoded_image_cache_size > 0:
            with self._decoded_image_cache_lock:
                cached = self._decoded_image_cache.get(key)
                if cached is not None:
                    self._decoded_image_cache.move_to_end(key)
                    return cached.copy()

        decoded = self._decode_replay_image(key)
        if self.decoded_image_cache_size > 0:
            with self._decoded_image_cache_lock:
                existing = self._decoded_image_cache.get(key)
                if existing is None:
                    self._decoded_image_cache[key] = decoded
                    self._decoded_image_cache.move_to_end(key)
                    while (
                        len(self._decoded_image_cache) > self.decoded_image_cache_size
                    ):
                        self._decoded_image_cache.popitem(last=False)
                else:
                    decoded = existing
                    self._decoded_image_cache.move_to_end(key)
        return decoded.copy()

    def _prepare_replay_tensor(self, path: str) -> torch.Tensor:
        key = os.path.abspath(path)
        with self._replay_tensor_cache_lock:
            cached = self._replay_tensor_cache.get(key)
        if cached is not None:
            return cached

        tensor = self.benchmark.prepare_replay_tensor(self._load_replay_image(key)).to(
            dtype=torch.float32
        )
        if self.device.type == "cuda" and not tensor.is_pinned():
            tensor = tensor.pin_memory()
        with self._replay_tensor_cache_lock:
            return self._replay_tensor_cache.setdefault(key, tensor)

    def _prepare_replay_tensors(self, paths: Sequence[str]) -> list[torch.Tensor]:
        if self._decode_executor is None or len(paths) < 2:
            return [self._prepare_replay_tensor(path) for path in paths]
        return list(self._decode_executor.map(self._prepare_replay_tensor, paths))

    def _synchronize_replay_tensor_cache(self) -> None:
        """Pre-transform every retained path and discard evicted cache entries."""
        retained_paths = {
            os.path.abspath(str(record.image_path))
            for record in self.replay_buffer.records()
            if record.image is None and record.image_path is not None
        }
        with self._replay_tensor_cache_lock:
            self._replay_tensor_cache = {
                path: tensor
                for path, tensor in self._replay_tensor_cache.items()
                if path in retained_paths
            }
        missing = [
            path for path in retained_paths if path not in self._replay_tensor_cache
        ]
        self._prepare_replay_tensors(missing)
        # Tensor-backed replay no longer needs a second full decoded-PIL cache.
        with self._decoded_image_cache_lock:
            self._decoded_image_cache.clear()

    @classmethod
    def _distillation_cache_key(cls, domain_id: int, image_id: Any) -> tuple[int, Any]:
        return int(domain_id), cls._image_id(image_id)

    def _build_distillation_target_cache(self, *, teacher_task_id: int) -> int:
        """Cache RoI-logit targets from the just-completed task model."""
        self._distillation_target_cache.clear()
        self._flipped_teacher_boxes.clear()
        if not self.distillation_enabled or not self.replay_enabled:
            return 0
        records = self.replay_buffer.records()
        if not records:
            return 0

        batch_size = max(
            1, int(self.config.get("training", {}).get("eval_batch_size", 2))
        )
        starts = range(0, len(records), batch_size)
        for start in self._progress(
            starts,
            desc=(
                f"CLAD-D teacher target cache | after domain {int(teacher_task_id) + 1}"
            ),
            unit="batch",
            leave=False,
        ):
            batch_records = records[start : start + batch_size]
            images: list[torch.Tensor] = []
            for record in batch_records:
                if record.image is not None:
                    image = (
                        record.image.detach()
                        if torch.is_tensor(record.image)
                        else self.benchmark.prepare_replay_tensor(record.image)
                    )
                elif record.image_path is not None:
                    image = self._prepare_replay_tensor(str(record.image_path))
                else:
                    raise ValueError("Replay record has neither an image nor a path")
                images.append(image.to(self.device, non_blocking=True))
            with self._autocast():
                extracted = self.model.extract_replay_distillation_targets(
                    images,
                    max_proposals_per_image=self.distillation_max_proposals,
                )
            if len(extracted) != len(batch_records):
                raise RuntimeError(
                    "Teacher target extraction returned the wrong batch size"
                )
            for record, target in zip(batch_records, extracted):
                key = self._distillation_cache_key(record.domain_id, record.image_id)
                boxes = target["boxes"].detach().to(device="cpu", dtype=torch.float32)
                logits = (
                    target["logits"]
                    .detach()
                    .to(device="cpu", dtype=self.distillation_cache_dtype)
                )
                if self.device.type == "cuda":
                    boxes = boxes.pin_memory()
                    logits = logits.pin_memory()
                self._distillation_target_cache[key] = {
                    "boxes": boxes,
                    "logits": logits,
                }

        expected_keys = {
            self._distillation_cache_key(record.domain_id, record.image_id)
            for record in records
        }
        if set(self._distillation_target_cache) != expected_keys:
            raise RuntimeError("Teacher target cache does not cover the replay buffer")
        return len(self._distillation_target_cache)

    def _teacher_targets_for_replay_batch(
        self, replay_batch: Mapping[str, Any]
    ) -> list[Dict[str, torch.Tensor]]:
        targets: list[Dict[str, torch.Tensor]] = []
        for domain_id, image_id, flipped in zip(
            replay_batch["domain_ids"].tolist(),
            replay_batch["image_ids"],
            replay_batch["horizontal_flip_flags"],
        ):
            key = self._distillation_cache_key(domain_id, image_id)
            cached = self._distillation_target_cache.get(key)
            if cached is None:
                raise RuntimeError(
                    "Replay image has no target from the previous task teacher"
                )
            boxes = cached["boxes"]
            if bool(flipped):
                flipped_boxes = self._flipped_teacher_boxes.get(key)
                if flipped_boxes is None:
                    flipped_boxes = boxes.clone()
                    flipped_boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
                    if self.device.type == "cuda" and not flipped_boxes.is_pinned():
                        flipped_boxes = flipped_boxes.pin_memory()
                    self._flipped_teacher_boxes[key] = flipped_boxes
                boxes = flipped_boxes
            targets.append({"boxes": boxes, "logits": cached["logits"]})
        if len(targets) != len(replay_batch["images"]):
            raise RuntimeError("Replay teacher targets do not align with the batch")
        return targets

    @staticmethod
    def _image_id(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("Detection image_id must contain one value")
            return value.detach().cpu().item()
        if hasattr(value, "item"):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        return value

    def _official_replay_sources(self, task_id: int) -> Dict[Any, Dict[str, Any]]:
        cached = self._replay_source_cache.get(task_id)
        if cached is not None:
            return cached
        dataset = self.benchmark.train_datasets[task_id]
        official = getattr(dataset, "official_dataset", dataset)
        ids = getattr(official, "ids", None)
        annotations = getattr(official, "img_annotations", None)
        image_folder = getattr(official, "img_folder", None)
        load_target = getattr(official, "_load_target", None)
        sources: Dict[Any, Dict[str, Any]] = {}
        if (
            ids is not None
            and annotations is not None
            and image_folder is not None
            and callable(load_target)
        ):
            for index, raw_image_id in enumerate(ids):
                image_id = self._image_id(raw_image_id)
                sources[image_id] = {
                    "image_path": os.path.join(
                        image_folder, annotations[raw_image_id]["file_name"]
                    ),
                    "index": index,
                    "load_target": load_target,
                }
        self._replay_source_cache[task_id] = sources
        return sources

    def _admit_completed_domain(self, task_id: int) -> Dict[str, Any]:
        """Consider each completed-domain image for replay exactly once.

        Admission happens only after the task's final controller outcome has
        been measured. Consequently, replay used while learning task ``t``
        contains only records from domains strictly earlier than ``t``.
        """
        before_size = len(self.replay_buffer)
        before_counts = self.replay_buffer.domain_counts()
        if not self.replay_enabled:
            return {
                "considered": 0,
                "retained_from_task": 0,
                "size_before": before_size,
                "size_after": before_size,
                "domain_counts_before": before_counts,
                "domain_counts_after": before_counts,
            }

        sources = self._official_replay_sources(int(task_id))
        if not sources:
            raise RuntimeError(
                "Task-boundary replay admission requires official CLAD-D "
                "image IDs, targets, and image paths"
            )
        considered = 0
        candidates = []
        for image_id, source in sources.items():
            if self.replay_buffer.has_seen(domain_id=task_id, image_id=image_id):
                continue
            candidates.append(
                {
                    "image_id": image_id,
                    "domain_id": task_id,
                    "target": source["load_target"](source["index"]),
                    "image_path": source["image_path"],
                    "metadata": {
                        "domain_id": int(task_id),
                        "refinement_scored": False,
                    },
                }
            )
            considered += 1

        if self.class_balanced_replay_admission:
            self.replay_buffer.admit_balanced_candidates(
                candidates,
                balance_strength=self.balanced_replay_sampling_strength,
                class_balance_beta=self.class_balance_beta,
                max_sample_weight=self.balanced_replay_sampling_max_weight,
                utility_sampling_strength=self.utility_sampling_strength,
            )
        else:
            self.replay_buffer.add_many(candidates)

        after_counts = self.replay_buffer.domain_counts()
        self._synchronize_replay_tensor_cache()
        return {
            "considered": considered,
            "retained_from_task": int(after_counts.get(int(task_id), 0)),
            "size_before": before_size,
            "size_after": len(self.replay_buffer),
            "domain_counts_before": before_counts,
            "domain_counts_after": after_counts,
            "class_image_counts_after": self.replay_buffer.class_image_counts(),
            "class_object_counts_after": self.replay_buffer.class_object_counts(),
            "object_size_counts_after": self.replay_buffer.object_size_counts(),
        }

    @staticmethod
    def _record_candidate(record: Any) -> Dict[str, Any]:
        candidate: Dict[str, Any] = {
            "image_id": record.image_id,
            "domain_id": record.domain_id,
            "target": record.target,
            "selection_score": record.selection_score,
            "metadata": record.metadata or {},
        }
        if record.image is not None:
            candidate["image"] = record.image
        else:
            candidate["image_path"] = record.image_path
        return candidate

    def _refinement_candidates(self) -> list[Dict[str, Any]]:
        """Return a rotating subset of retained *old-domain* records.

        Refinement updates detection difficulty scores used by replay sampling;
        it cannot admit the current task early or access non-retained old images.
        This keeps the rehearsal state strictly within the 250-image buffer.
        """
        if self.refinement_candidate_count == 0 or self.current_task_id <= 0:
            return []
        eligible = [
            record
            for record in self.replay_buffer.records()
            if int(record.domain_id) < self.current_task_id
        ]
        if not eligible:
            return []
        count = min(self.refinement_candidate_count, len(eligible))
        cursor = self._refinement_cursor % len(eligible)
        selected = [
            eligible[(cursor + offset) % len(eligible)] for offset in range(count)
        ]
        self._refinement_cursor = (cursor + count) % len(eligible)
        return [self._record_candidate(record) for record in selected]

    @staticmethod
    def _refinement_score(
        prediction: Mapping[str, torch.Tensor],
        target: Mapping[str, torch.Tensor],
        roi_probabilities: torch.Tensor,
        *,
        rare_class_ids: Sequence[int],
        entropy_weight: float,
        localization_weight: float,
        rare_class_weight: float,
    ) -> tuple[float, float, float, float]:
        probabilities = torch.as_tensor(roi_probabilities, dtype=torch.float32)
        if (
            probabilities.ndim == 2
            and probabilities.shape[1] == 7
            and len(probabilities)
        ):
            foreground = probabilities[:, 1:].clamp_min(1e-8)
            foreground_confidence = foreground.sum(dim=1)
            conditional = foreground / foreground_confidence[:, None].clamp_min(1e-8)
            proposal_entropy = -(conditional * conditional.log()).sum(dim=1) / math.log(
                6.0
            )
            keep = min(100, len(proposal_entropy))
            selected = foreground_confidence.topk(keep).indices
            weights = foreground_confidence[selected]
            entropy = (
                proposal_entropy[selected] * weights
            ).sum() / weights.sum().clamp_min(1e-8)
            entropy_value = float(entropy.cpu())
        else:
            entropy_value = 0.0

        ground_truth_boxes = torch.as_tensor(target["boxes"], dtype=torch.float32)
        ground_truth_labels = torch.as_tensor(target["labels"], dtype=torch.int64)
        predicted_boxes = torch.as_tensor(
            prediction.get("boxes", torch.empty((0, 4))), dtype=torch.float32
        )
        predicted_labels = torch.as_tensor(
            prediction.get("labels", torch.empty((0,), dtype=torch.int64)),
            dtype=torch.int64,
        )
        if ground_truth_boxes.numel() and predicted_boxes.numel():
            pairwise_iou = box_iou(ground_truth_boxes, predicted_boxes)
            same_class = ground_truth_labels[:, None] == predicted_labels[None, :]
            best_iou = pairwise_iou.masked_fill(~same_class, -1.0).max(dim=1).values
            best_iou = best_iou.clamp_min(0.0)
            localization_value = float((1.0 - best_iou).mean().item())
        elif ground_truth_boxes.numel():
            localization_value = 1.0
        else:
            localization_value = 0.0
        rare_ids = torch.as_tensor(tuple(rare_class_ids), dtype=torch.int64)
        rare_value = (
            float(torch.isin(ground_truth_labels, rare_ids).float().mean().item())
            if rare_ids.numel() and ground_truth_labels.numel()
            else 0.0
        )
        utility = (
            entropy_weight * entropy_value
            + localization_weight * localization_value
            + rare_class_weight * rare_value
        )
        return (
            float(utility),
            float(entropy_value),
            float(localization_value),
            float(rare_value),
        )

    @torch.inference_mode()
    def _score_refinement_candidates(
        self, candidates: Sequence[Mapping[str, Any]]
    ) -> list[Dict[str, Any]]:
        if not candidates:
            return []
        prepared: list[tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]] = []
        transform = getattr(self.benchmark, "eval_transform", None)
        for raw_candidate in candidates:
            candidate = deepcopy(dict(raw_candidate))
            image = candidate.get("image")
            already_transformed = False
            if image is None:
                image_path = candidate.get("image_path")
                if image_path is None:
                    raise ValueError("A replay refinement candidate has no image")
                image = self._prepare_replay_tensor(str(image_path))
                already_transformed = True
            elif torch.is_tensor(image):
                image = image.detach().cpu().clone()
            target = _cpu_detection_dict(candidate["target"])
            if transform is not None and not already_transformed:
                image, target = transform(image, target)
            if not torch.is_tensor(image) or image.ndim != 3:
                raise TypeError(
                    "Replay refinement transform must return a [C,H,W] tensor"
                )
            prepared.append((image, target, candidate))

        batch_size = max(
            1, int(self.config.get("training", {}).get("eval_batch_size", 2))
        )
        scored: list[Dict[str, Any]] = []
        rare_class_ids = self._rare_class_ids(self.current_task_id)
        was_training = self.model.training
        self.model.eval()
        try:
            starts = range(0, len(prepared), batch_size)
            for start in self._progress(
                starts,
                desc=(
                    "CLAD-D replay refinement | "
                    f"domain {self.current_task_id + 1}/{self.benchmark.num_tasks}"
                ),
                unit="batch",
                leave=False,
            ):
                batch = prepared[start : start + batch_size]
                device_images = [item[0].to(self.device) for item in batch]
                with self._autocast():
                    refinement_signals = getattr(
                        self.model, "predict_with_refinement_signals", None
                    )
                    if not callable(refinement_signals):
                        raise RuntimeError(
                            "CARL-D detector must expose pre-NMS refinement signals"
                        )
                    predictions, roi_probabilities, feature_vectors = (
                        refinement_signals(device_images)
                    )
                for prediction, probabilities, feature, (
                    _,
                    transformed_target,
                    candidate,
                ) in zip(predictions, roi_probabilities, feature_vectors, batch):
                    utility, entropy, localization, rare_gain = self._refinement_score(
                        _cpu_detection_dict(prediction),
                        transformed_target,
                        probabilities.detach().cpu(),
                        rare_class_ids=rare_class_ids,
                        entropy_weight=self.refinement_entropy_weight,
                        localization_weight=self.refinement_localization_weight,
                        rare_class_weight=self.refinement_rare_class_weight,
                    )
                    candidate["selection_score"] = utility
                    metadata = dict(candidate.get("metadata", {}))
                    metadata.update(
                        refinement_scored=True,
                        refinement_optimizer_step=int(self.optimizer_step),
                        refinement_foreground_entropy=entropy,
                        refinement_localization_error=localization,
                        refinement_rare_class_gain=rare_gain,
                        refinement_feature=feature.detach().cpu(),
                    )
                    candidate["metadata"] = metadata
                    scored.append(candidate)
        finally:
            self.model.train(was_training)
        return scored

    def _refine_replay_buffer(self) -> tuple[bool, int]:
        candidates = self._refinement_candidates()
        scored = self._score_refinement_candidates(candidates)
        changed = self.replay_buffer.refine_with_candidates(
            scored,
            feature_redundancy_weight=self.refinement_feature_redundancy_weight,
        )
        return changed, len(scored)

    def _prefetch_replay_records(
        self,
    ) -> tuple[list[DetectionReplayRecord], dict[str, Future[torch.Tensor]]]:
        """Draw once with the buffer-local RNG; prefetch deterministic tensors only."""
        if not self.replay_enabled or len(self.replay_buffer) == 0:
            return [], {}
        records = self.replay_buffer.sample_records(
            self.replay_batch_size,
            domain_class_balanced=self.balanced_replay_sampling,
            batch_aware_sampling=self.batch_aware_replay_sampling,
            balance_strength=self.balanced_replay_sampling_strength,
            max_sample_weight=self.balanced_replay_sampling_max_weight,
            class_balance_beta=self.class_balance_beta,
            utility_sampling_strength=self.utility_sampling_strength,
            max_domain_id_exclusive=self.current_task_id,
        )
        futures = {}
        if self._decode_executor is not None:
            for record in records:
                path = record.image_path
                if record.image is None and path is not None and path not in futures:
                    with self._replay_tensor_cache_lock:
                        cached = os.path.abspath(path) in self._replay_tensor_cache
                    if cached:
                        continue
                    futures[path] = self._decode_executor.submit(
                        self._prepare_replay_tensor, path
                    )
        return records, futures

    def _sample_replay_batch(
        self,
        prepared: Optional[
            tuple[list[DetectionReplayRecord], dict[str, Future[torch.Tensor]]]
        ] = None,
        *,
        transfer_to_device: bool = True,
    ) -> Optional[Dict[str, Any]]:
        records, futures = (
            self._prefetch_replay_records() if prepared is None else prepared
        )

        def load_selected(paths):
            return [
                futures[path].result()
                if path in futures
                else self._prepare_replay_tensor(path)
                for path in paths
            ]

        batch = self.replay_buffer.materialize_records(
            records,
            image_loader=self._prepare_replay_tensor,
            batch_image_loader=load_selected,
            transform=self.benchmark.transform_cached_replay,
            device=self.device if transfer_to_device else None,
        )
        if batch is not None and bool(
            (batch["domain_ids"] >= self.current_task_id).any()
        ):
            raise RuntimeError(
                "Replay sampler returned a current or future domain record"
            )
        return batch

    def _record_optimizer_step(self, optimizer, args, kwargs) -> None:
        # For non-fused SGD, GradScaler does not call step on overflow.
        self._optimizer_did_step = True

    def _optimizer_update(self) -> bool:
        self._optimizer_did_step = False
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        update_was_skipped = not self._optimizer_did_step
        self.optimizer.zero_grad(set_to_none=True)
        if update_was_skipped:
            return False
        self._advance_warmup()
        self.optimizer_step += 1
        self.task_optimizer_step += 1
        return True

    def _train_epoch(
        self,
        train_loader: Iterable[Any],
        *,
        epoch: Optional[int] = None,
    ) -> Dict[str, float]:
        self.model.train()
        self._enforce_stagewise_module_modes()
        self.optimizer.zero_grad(set_to_none=True)
        batch_count = 0
        statistics = torch.zeros(7, device=self.device, dtype=torch.float64)
        ema = torch.tensor(
            [
                self.loss_ema,
                self.recognition_loss_ema,
                self.localization_loss_ema,
            ],
            device=self.device,
            dtype=torch.float64,
        )
        ema_updates = int(self.loss_ema_updates)

        epoch_number = self.current_epoch + 1 if epoch is None else int(epoch)
        progress = self._progress(
            train_loader,
            desc=(
                "CLAD-D train | "
                f"domain {self.current_task_id + 1}/{self.benchmark.num_tasks} | "
                f"epoch {epoch_number}/{self.epochs_per_task}"
            ),
            unit="batch",
            leave=False,
        )
        for images, targets in progress:
            prepared_replay = self._prefetch_replay_records()
            current_loss, current_loss_terms = self._detector_losses(images, targets)
            # Backpropagate before the replay forward to avoid retaining both
            # detector activation graphs and to keep mutable compiled-module
            # state out of a later combined backward. Both gradients still
            # accumulate into the same optimizer step.
            self.scaler.scale(current_loss).backward()

            replay_loss = torch.zeros((), device=self.device)
            distillation_loss = torch.zeros((), device=self.device)
            distillation_matches = torch.zeros((), device=self.device)
            # Keep CPU augmentation here, AFTER the current backward, so the
            # global RNG order is identical. Only copies use a separate stream;
            # they may overlap the remaining queued current-batch GPU work.
            replay_transfer = None
            if self._replay_transfer_stream is not None:
                replay_batch = self._sample_replay_batch(
                    prepared_replay, transfer_to_device=False
                )
                if replay_batch is not None:
                    replay_transfer = _CudaReplayTransfer(
                        replay_batch, self.device, self._replay_transfer_stream
                    )
                    replay_batch = replay_transfer.wait()
            else:
                replay_batch = self._sample_replay_batch(prepared_replay)
            if replay_batch is not None:
                if self.distillation_enabled:
                    (
                        replay_loss,
                        _,
                        distillation_loss,
                        distillation_matches,
                    ) = self._replay_losses_with_distillation(replay_batch)
                else:
                    replay_loss, _ = self._detector_losses(
                        replay_batch["images"], replay_batch["targets"]
                    )
                replay_objective = (
                    self.replay_loss_weight * replay_loss
                    + self.distillation_loss_weight * distillation_loss
                )
                self.scaler.scale(replay_objective).backward()
            combined = (
                current_loss.detach()
                + self.replay_loss_weight * replay_loss.detach()
                + self.distillation_loss_weight * distillation_loss.detach()
            )

            zero_loss = torch.zeros((), device=self.device)
            recognition_loss = current_loss_terms.get("loss_classifier", zero_loss)
            localization_loss = current_loss_terms.get(
                "loss_box_reg", zero_loss
            ) + current_loss_terms.get("loss_rpn_box_reg", zero_loss)
            values = torch.stack(
                (
                    combined.detach(),
                    current_loss.detach(),
                    replay_loss.detach(),
                    distillation_loss.detach(),
                    distillation_matches.detach(),
                    recognition_loss.detach(),
                    localization_loss.detach(),
                )
            ).to(dtype=torch.float64)
            if ema_updates == 0:
                ema.copy_(values[[0, 5, 6]])
            else:
                ema.mul_(0.95).add_(values[[0, 5, 6]], alpha=0.05)
            ema_updates += 1
            statistics.add_(values)
            batch_count += 1

            if batch_count == 1 or batch_count % 20 == 0:
                progress.set_postfix(
                    {
                        "lr": f"{self._display_learning_rate():.2e}",
                        "buffer": (
                            f"{len(self.replay_buffer)}/"
                            f"{self.replay_buffer.current_capacity}"
                        ),
                        "replay_w": f"{self.replay_loss_weight:g}",
                    },
                    refresh=False,
                )

            self._optimizer_update()

        denominator = max(1, batch_count)
        packed = torch.cat((statistics, ema)).cpu().tolist()
        self.loss_ema = float(packed[7])
        self.recognition_loss_ema = float(packed[8])
        self.localization_loss_ema = float(packed[9])
        self.loss_ema_updates = ema_updates
        return {
            "current_detection_loss": float(packed[1]) / denominator,
            "replay_detection_loss": float(packed[2]) / denominator,
            "replay_distillation_loss": float(packed[3]) / denominator,
            "distillation_matches_per_batch": float(packed[4]) / denominator,
            "optimizer_steps": float(self.task_optimizer_step),
        }

    @torch.inference_mode()
    def _collect_domain_outputs(
        self,
        loader: Iterable[Any],
        *,
        progress_description: str = "CLAD-D evaluation",
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], int, int]:
        was_training = self.model.training
        self.model.eval()
        predictions: list[Dict[str, Any]] = []
        targets_cpu: list[Dict[str, Any]] = []
        matched_ground_truth = torch.zeros((), dtype=torch.int64, device=self.device)
        total_ground_truth = 0
        try:
            progress = self._progress(
                loader,
                desc=progress_description,
                unit="batch",
                leave=False,
            )
            for images, targets in progress:
                device_images = [
                    image.to(self.device, non_blocking=True) for image in images
                ]
                device_targets = [
                    _to_device(dict(target), self.device) for target in targets
                ]
                with self._autocast():
                    proposal_recall = getattr(
                        self.model, "predict_with_rpn_recall", None
                    )
                    if callable(proposal_recall):
                        recall_options = (
                            {"return_tensor_counts": True}
                            if isinstance(self.model, CARLDDetector)
                            else {}
                        )
                        outputs, matched, total = proposal_recall(
                            device_images, device_targets,
                            iou_threshold=0.5,
                            **recall_options,
                        )
                    else:
                        outputs = self.model(device_images)
                        matched = 0
                        total = 0
                        # Detectors without exposed RPN proposals use final-box
                        # recall; CARLDDetector uses exact post-NMS RPN recall.
                        for output, target in zip(outputs, device_targets):
                            ground_truth = target["boxes"]
                            total += len(ground_truth)
                            predicted = output.get(
                                "boxes", ground_truth.new_empty((0, 4))
                            )
                            if len(ground_truth) and len(predicted):
                                matched += int(
                                    (
                                        _box_iou_matrix(ground_truth, predicted)
                                        .max(1)
                                        .values
                                        >= 0.5
                                    )
                                    .sum()
                                    .item()
                                )
                predictions.extend(_cpu_detection_batch(outputs))
                targets_cpu.extend(_cpu_detection_dict(target) for target in targets)
                matched_ground_truth += matched
                total_ground_truth += int(total)
        finally:
            self.model.train(was_training)
        return (
            predictions,
            targets_cpu,
            int(matched_ground_truth.item()),
            total_ground_truth,
        )

    def evaluate_domains(
        self,
        loaders: Mapping[int, Iterable[Any]],
        *,
        phase: str = "evaluation",
        evaluator: Optional[Callable[..., Any]] = None,
    ) -> Any:
        predictions: Dict[int, Sequence[Mapping[str, torch.Tensor]]] = {}
        targets: Dict[int, Sequence[Mapping[str, torch.Tensor]]] = {}
        matched_ground_truth = 0
        total_ground_truth = 0
        for domain_id, loader in loaders.items():
            (
                domain_predictions,
                domain_targets,
                domain_matched,
                domain_total,
            ) = self._collect_domain_outputs(
                loader,
                progress_description=(
                    f"CLAD-D {phase} | "
                    f"domain {int(domain_id) + 1}/{self.benchmark.num_tasks}"
                ),
            )
            predictions[int(domain_id)] = domain_predictions
            targets[int(domain_id)] = domain_targets
            matched_ground_truth += domain_matched
            total_ground_truth += domain_total
        self.latest_rpn_recall = (
            matched_ground_truth / total_ground_truth if total_ground_truth else 0.0
        )
        selected_evaluator = evaluator or self.domain_evaluator
        return selected_evaluator(
            predictions,
            targets,
            foreground_class_ids=self.foreground_class_ids,
        )

    def evaluate_seen_domains(self, task_id: int) -> DomainAP50Summary:
        loaders: Dict[int, Iterable[Any]] = {}
        for domain_id in range(task_id + 1):
            loader = self._validation_loader_cache.get(domain_id)
            if loader is None:
                loader = self.benchmark.get_validation_dataloader(domain_id)
                self._validation_loader_cache[domain_id] = loader
            loaders[domain_id] = loader
        return self.evaluate_domains(loaders, phase="validation")

    def _incoming_domain_state(self, task_id: int) -> DomainAP50Summary:
        """Evaluate only the incoming domain and reuse unchanged old results."""
        loader = self._validation_loader_cache.get(task_id)
        if loader is None:
            loader = self.benchmark.get_validation_dataloader(task_id)
            self._validation_loader_cache[task_id] = loader
        incoming = self.evaluate_domains({task_id: loader}, phase="validation")
        if task_id == 0:
            return incoming
        if self.pending_old_domain_results is None:
            raise RuntimeError("Incoming-domain state is missing old-domain results")

        per_domain: Dict[int, AP50Result] = {}
        for raw_domain, raw in self.pending_old_domain_results.items():
            per_domain[int(raw_domain)] = AP50Result(
                map50=float(raw["map50"]),
                per_class_ap50={
                    int(class_id): float(value)
                    for class_id, value in raw["per_class_ap50"].items()
                },
                target_count_per_class={
                    int(class_id): int(value)
                    for class_id, value in raw["target_count_per_class"].items()
                },
                num_images=int(raw["num_images"]),
            )
        per_domain.update(incoming.per_domain)

        def finite_mean(values: Sequence[float]) -> float:
            finite = [float(value) for value in values if math.isfinite(float(value))]
            return sum(finite) / len(finite) if finite else math.nan

        return DomainAP50Summary(
            per_domain=per_domain,
            equal_domain_map50=finite_mean(
                [result.map50 for result in per_domain.values()]
            ),
            per_class_equal_domain_ap50={
                class_id: finite_mean(
                    [result.per_class_ap50[class_id] for result in per_domain.values()]
                )
                for class_id in self.foreground_class_ids
            },
        )

    def evaluate_test(self) -> Dict[str, Any]:
        loaders = {
            domain_id: loader
            for domain_id, loader in enumerate(
                self.benchmark.get_all_test_dataloaders()
            )
        }
        environment = {}

        def evaluate_test_outputs(predictions, targets, *, foreground_class_ids):
            result = evaluate_domains_ap50(
                predictions, targets, foreground_class_ids=foreground_class_ids
            )
            environment.update(evaluate_environment_ap50(
                predictions, targets, self.benchmark.get_test_image_metadata(),
                foreground_class_ids=foreground_class_ids,
            ))
            return result

        result = self.evaluate_domains(
            loaders,
            phase="test",
            evaluator=evaluate_test_outputs,
        )
        return {
            **result.as_dict(),
            "model_name": "CARL-FS" if self.fixed_schedule else "CARL-D",
            "per_environment": environment,
            "metric": (
                "CLAD-D 2023 COCO 101-point interpolated bounding-box AP "
                "at IoU 0.50, then equal-domain mean"
            ),
            "foreground_class_ids": list(self.foreground_class_ids),
        }

    def _training_class_counts(self, task_id: int) -> Dict[int, int]:
        counts = {class_id: 0 for class_id in self.foreground_class_ids}
        for seen_task in range(task_id + 1):
            cached = self._task_class_count_cache.get(seen_task)
            if cached is None:
                info = self.benchmark.get_task_info(seen_task)
                cached = {
                    int(class_id): int(count)
                    for class_id, count in info["class_object_counts"].items()
                }
                self._task_class_count_cache[seen_task] = cached
            for class_id in counts:
                counts[class_id] += cached.get(class_id, 0)
        return counts

    def _rare_class_ids(self, task_id: int) -> tuple[int, ...]:
        counts = self._training_class_counts(task_id)
        present = sorted(
            (count, class_id) for class_id, count in counts.items() if count > 0
        )
        return tuple(class_id for _, class_id in present[:2])

    def _set_class_balanced_roi_weights(self, task_id: int) -> None:
        """Update mild effective-number weights from all seen train domains."""
        if not self.class_balanced_roi_loss:
            return
        counts = torch.zeros(7, dtype=torch.float64)
        get_task_info = getattr(self.benchmark, "get_task_info", None)
        if not callable(get_task_info):
            self.model.set_roi_class_weights(torch.ones(7))
            return
        for seen_task in range(task_id + 1):
            cached = self._task_class_count_cache.get(seen_task)
            if cached is None:
                info = get_task_info(seen_task)
                cached = {
                    int(class_id): int(count)
                    for class_id, count in info["class_object_counts"].items()
                }
                self._task_class_count_cache[seen_task] = cached
            for class_id in self.foreground_class_ids:
                counts[class_id] += cached.get(class_id, 0)

        weights = torch.ones(7, dtype=torch.float64)
        present = counts[1:] > 0
        if present.any():
            class_counts = counts[1:][present]
            beta = self.class_balance_beta
            effective = (1.0 - beta) / (
                1.0 - torch.pow(torch.full_like(class_counts, beta), class_counts)
            )
            effective /= effective.mean().clamp_min(torch.finfo(effective.dtype).eps)
            effective.clamp_(
                min=1.0 / self.class_balance_max_weight,
                max=self.class_balance_max_weight,
            )
            weights[1:][present] = effective
        weights[0] = 1.0
        self.model.set_roi_class_weights(weights.to(dtype=torch.float32))

    def _controller_metrics(
        self, task_id: int, result: DomainAP50Summary
    ) -> Dict[str, Any]:
        official_domain_values = {
            int(domain): _finite_or_zero(domain_result.map50)
            for domain, domain_result in result.per_domain.items()
        }
        rare_class_ids = self._rare_class_ids(task_id)
        for domain, domain_result in result.per_domain.items():
            counts = dict(domain_result.target_count_per_class)
            previous = self._feedback_annotation_counts.setdefault(int(domain), counts)
            if previous != counts:
                raise ValueError(
                    "Controller validation annotation support changed within the run"
                )
        domain_values, rare_class_ap50, support_weights = support_aware_ap(
            result, rare_class_ids, self.feedback_support_scale
        )
        old_domain_values = {
            domain: value for domain, value in domain_values.items() if domain < task_id
        }
        forgetting_values = [
            max(0.0, self.best_domain_ap50.get(domain, value) - value)
            for domain, value in old_domain_values.items()
        ]
        forgetting = (
            sum(forgetting_values) / len(forgetting_values)
            if forgetting_values
            else 0.0
        )
        for domain, value in domain_values.items():
            self.best_domain_ap50[domain] = max(
                self.best_domain_ap50.get(domain, value), value
            )

        manifest = self.model.architecture_manifest()
        growth = {
            group: self.model.adapter_parameter_count(group)
            / self.parameter_growth_budget
            for group in ("backbone", "fpn", "roi")
        }
        seen_domain_ap50 = sum(domain_values.values()) / max(1, len(domain_values))
        old_domain_ap50 = (
            sum(old_domain_values.values()) / len(old_domain_values)
            if old_domain_values
            else 0.0
        )

        def cooldown_ratio(last_step: Optional[int], duration: int) -> float:
            if last_step is None or duration <= 0:
                return 0.0
            elapsed = self.controller.decision_steps - last_step
            return max(0.0, duration - elapsed) / duration

        return {
            "task_id": int(task_id),
            "optimizer_step": int(self.optimizer_step),
            "epoch": int(self.current_epoch),
            "seen_domain_ap50": seen_domain_ap50,
            "feedback_convention": "support_aware_v1",
            "feedback_support_scale": self.feedback_support_scale,
            "feedback_support_weights": support_weights,
            "support_aware_per_domain_ap50": domain_values,
            "official_equal_domain_map50": float(result.equal_domain_map50),
            "official_per_domain_ap50": official_domain_values,
            "old_domain_ap50": old_domain_ap50,
            "current_domain_ap50": domain_values.get(task_id, 0.0),
            "worst_domain_ap50": min(domain_values.values(), default=0.0),
            "forgetting": forgetting,
            "rare_class_ap50": rare_class_ap50,
            "rare_class_ids": list(rare_class_ids),
            "rpn_recall": float(self.latest_rpn_recall),
            "loss_ema": float(self.loss_ema),
            "recognition_loss_ema": float(self.recognition_loss_ema),
            "localization_loss_ema": float(self.localization_loss_ema),
            "buffer_fill_ratio": self.replay_buffer.get_fill_ratio(),
            "buffer_capacity_ratio": self.replay_buffer.get_capacity_ratio(),
            "replay_loss_weight": float(self.replay_loss_weight),
            "replay_loss_weight_ratio": (
                self._replay_loss_weight_index()
                / max(1, len(self.replay_loss_weight_levels) - 1)
            ),
            "parameter_growth_ratio": max(
                0.0,
                (self.model.get_num_parameters() - self.base_parameter_count)
                / self.parameter_growth_budget,
            ),
            "task_progress": self.current_epoch / self.epochs_per_task,
            "backbone_frozen": "backbone" in manifest["frozen_groups"],
            "freeze_wait_epochs": max(
                0, self.freeze_available_at_epoch - self._stream_epoch()
            ),
            "replay_weight_wait_epochs": max(
                0, self.replay_weight_available_at_epoch - self._stream_epoch()
            ),
            "expansion_ratios": growth,
            "structural_expansion_cooldown_ratio": cooldown_ratio(
                self.last_structural_expansion_decision_step,
                self.structural_expansion_cooldown_decisions,
            ),
        }

    def _replay_loss_weight_index(self) -> int:
        for index, value in enumerate(self.replay_loss_weight_levels):
            if math.isclose(self.replay_loss_weight, value):
                return index
        raise RuntimeError("Current replay loss weight is not a configured level")

    def _replay_snapshot(self) -> Dict[str, Any]:
        return {
            "size": len(self.replay_buffer),
            "capacity": self.replay_buffer.current_capacity,
            "domain_counts": self.replay_buffer.domain_counts(),
            "class_image_counts": self.replay_buffer.class_image_counts(),
        }

    def _stream_epoch(self) -> int:
        """Completed training epochs, continuous across domain boundaries."""
        return max(0, self.current_task_id) * self.epochs_per_task + self.current_epoch

    def valid_action_ids(self) -> list[int]:
        valid = [int(CARLDAction.NO_OP)]
        old_replay_available = self.replay_enabled and any(
            int(domain_id) < self.current_task_id and int(count) > 0
            for domain_id, count in self.replay_buffer.domain_counts().items()
        )
        if (
            old_replay_available
            and self._stream_epoch() >= self.replay_weight_available_at_epoch
        ):
            replay_level = self._replay_loss_weight_index()
            if replay_level < len(self.replay_loss_weight_levels) - 1:
                valid.append(int(CARLDAction.INCREASE_REPLAY_WEIGHT))
            if replay_level > 0:
                valid.append(int(CARLDAction.DECREASE_REPLAY_WEIGHT))
        if old_replay_available and self.refinement_candidate_count > 0:
            valid.append(int(CARLDAction.REFINE_BUFFER))

        frozen = self.model.architecture_manifest()["frozen_groups"]
        if (
            "backbone" not in frozen
            and self._stream_epoch() >= self.freeze_available_at_epoch
        ):
            valid.append(int(CARLDAction.FREEZE_BACKBONE))

        if self.adapters_enabled:
            current_parameters = self.model.get_num_parameters()
            expansion_on_cooldown = (
                not self.fixed_schedule
                and self.last_structural_expansion_decision_step is not None
                and self.controller.decision_steps
                - self.last_structural_expansion_decision_step
                < self.structural_expansion_cooldown_decisions
            )
            for action, group in EXPANSION_ACTION_GROUPS.items():
                if expansion_on_cooldown:
                    continue
                if group not in self.adapter_groups:
                    continue
                if (
                    self.model.adapter_generation_count(group)
                    >= self.adapter_generation_limits[group]
                ):
                    continue
                estimated = self.model.estimate_expansion_params(group)
                if current_parameters + estimated <= self.max_parameter_count:
                    valid.append(int(action))
        return valid

    def execute_action(
        self, action: int, *, applies_to_task_id: Optional[int] = None
    ) -> ActionExecutionEvent:
        action = int(action)
        event = ActionExecutionEvent(
            task_id=self.current_task_id,
            applies_to_task_id=(
                self.current_task_id
                if applies_to_task_id is None
                else int(applies_to_task_id)
            ),
            epoch=self.current_epoch,
            optimizer_step=self.optimizer_step,
            action_id=action,
            action_name=ACTION_NAMES.get(action, "unknown"),
            invalid=action not in self.valid_action_ids(),
            replay_loss_weight_before=float(self.replay_loss_weight),
            replay_before=self._replay_snapshot(),
        )
        if event.invalid:
            event.replay_loss_weight_after = float(self.replay_loss_weight)
            event.replay_after = self._replay_snapshot()
            self.action_events.append(event)
            return event

        if action == int(CARLDAction.NO_OP):
            pass
        elif action == int(CARLDAction.INCREASE_REPLAY_WEIGHT):
            level = self._replay_loss_weight_index()
            self.replay_loss_weight = self.replay_loss_weight_levels[level + 1]
            self.replay_weight_available_at_epoch = (
                self._stream_epoch() + self.replay_weight_freeze_epochs
            )
            event.changed = True
        elif action == int(CARLDAction.DECREASE_REPLAY_WEIGHT):
            level = self._replay_loss_weight_index()
            self.replay_loss_weight = self.replay_loss_weight_levels[level - 1]
            self.replay_weight_available_at_epoch = (
                self._stream_epoch() + self.replay_weight_freeze_epochs
            )
            event.changed = True
        elif action == int(CARLDAction.REFINE_BUFFER):
            event.changed, event.candidate_count = self._refine_replay_buffer()
        elif action == int(CARLDAction.FREEZE_BACKBONE):
            event.changed = self.model.freeze_group(
                "backbone",
                until_step=None,
                preserve_adapters=True,
            )
            event.duration_epochs = self.decision_frequency_epochs
            self.freeze_available_at_epoch = (
                self._stream_epoch()
                + event.duration_epochs
                + self.freeze_cooldown_epochs
            )
            event.until_task_epoch = (
                self.decision_frequency_epochs
                if event.applies_to_task_id != self.current_task_id
                else self.current_epoch + self.decision_frequency_epochs
            )
        elif action in EXPANSION_ACTION_GROUPS:
            group = EXPANSION_ACTION_GROUPS[action]
            event.added_parameters = self.model.expand_group(group)
            event.changed = True
            self._refresh_optimizer_after_expansion()
            self.last_structural_expansion_decision_step = int(
                self.controller.decision_steps
            )
        else:
            event.invalid = True
        event.replay_loss_weight_after = float(self.replay_loss_weight)
        event.replay_after = self._replay_snapshot()
        self.action_events.append(event)
        return event

    def _record_evaluation(
        self,
        *,
        task_id: int,
        phase: str,
        result: DomainAP50Summary,
        metrics: Mapping[str, Any],
    ) -> None:
        self.evaluation_history.append(
            {
                "task_id": task_id,
                "epoch": self.current_epoch,
                "optimizer_step": self.optimizer_step,
                "phase": phase,
                "result": result.as_dict(),
                "controller_metrics": deepcopy(dict(metrics)),
            }
        )

    def _decision_boundary(
        self,
        task_id: int,
        result: DomainAP50Summary,
        *,
        phase: str,
        terminal: bool,
        applies_to_task_id: Optional[int] = None,
        defer_successor: bool = False,
    ) -> None:
        metrics = self._controller_metrics(task_id, result)
        self._record_evaluation(
            task_id=task_id, phase=phase, result=result, metrics=metrics
        )

        if self.fixed_schedule:
            # No learned decisions or replay transitions. Retain old-domain
            # evaluation results for the next task's incoming-domain probe.
            self.pending_old_domain_results = (
                {int(d): r.as_dict() for d, r in result.per_domain.items()}
                if defer_successor else None
            )
            if phase == "task_start" and task_id > 0:
                for group in ("backbone", "fpn", "roi"):
                    action = next(
                        a for a, g in EXPANSION_ACTION_GROUPS.items() if g == group
                    )
                    if int(action) in self.valid_action_ids():
                        self.controller.record(self.execute_action(action))
            elif (
                phase in ("interval", "task_end") and not terminal
                and self.current_epoch % 2 == 0
                and int(CARLDAction.REFINE_BUFFER) in self.valid_action_ids()
            ):
                self.controller.record(self.execute_action(CARLDAction.REFINE_BUFFER))
            # Scheduled refinement includes nonterminal task boundaries.
            self.controller.decision_steps += 1
            return

        if defer_successor:
            if terminal or self.pending_decision is None:
                raise RuntimeError("Only a nonterminal pending action can be deferred")
            self.pending_outcome_metrics = deepcopy(metrics)
            self.pending_old_domain_results = {
                int(domain): domain_result.as_dict()
                for domain, domain_result in result.per_domain.items()
            }
            return

        next_valid = [] if terminal else self.valid_action_ids()
        if self.pending_decision is not None:
            reward_metrics = (
                metrics
                if self.pending_outcome_metrics is None
                else self.pending_outcome_metrics
            )
            self.controller.observe(
                self.pending_decision["metrics"],
                self.pending_decision["action"],
                reward_metrics,
                next_state_metrics=metrics,
                done=terminal,
                next_valid_action_ids=next_valid,
                invalid_action=bool(self.pending_decision["invalid"]),
            )
            self.pending_decision = None
            self.pending_outcome_metrics = None
            self.pending_old_domain_results = None

        if terminal:
            return
        action, _ = self.controller.select_action(metrics, next_valid)
        event = self.execute_action(action, applies_to_task_id=applies_to_task_id)
        self.controller.annotate_last_action_execution(event.to_dict(), action=action)
        self.pending_decision = {
            "metrics": deepcopy(metrics),
            "action": int(action),
            "invalid": bool(event.invalid),
            "applies_to_task_id": event.applies_to_task_id,
        }

    def train_task(self, task_id: int) -> Dict[str, Any]:
        if task_id != self.next_task_id:
            raise ValueError(
                f"Expected task {self.next_task_id}, received task {task_id}"
            )
        if task_id == 0 and (
            self.pending_decision is not None
            or self.pending_outcome_metrics is not None
            or self.pending_old_domain_results is not None
        ):
            raise RuntimeError("The first domain cannot start with a pending action")
        if task_id > 0 and not self.fixed_schedule and (
            self.pending_decision is None
            or self.pending_outcome_metrics is None
            or self.pending_old_domain_results is None
        ):
            raise RuntimeError(
                "A noninitial domain must continue a deferred full-stream transition"
            )

        self.current_task_id = task_id
        self.current_epoch = 0
        self.task_optimizer_step = 0
        self._configure_stagewise_protection(task_id)
        replay_images_available_during_task = len(self.replay_buffer)
        replay_domain_counts_during_task = self.replay_buffer.domain_counts()
        if any(
            int(domain_id) >= task_id for domain_id in replay_domain_counts_during_task
        ):
            raise RuntimeError(
                "Replay for a task may contain only completed earlier domains"
            )
        if (
            self.distillation_enabled
            and task_id > 0
            and len(self._distillation_target_cache) != len(self.replay_buffer)
        ):
            raise RuntimeError(
                "Task cannot start without teacher targets for every replay image"
            )
        replay_loss_weight_start = float(self.replay_loss_weight)
        self._set_class_balanced_roi_weights(task_id)
        check_model_health(self.model)

        loader = self.benchmark.get_task_dataloader(task_id, train=True)
        try:
            updates_per_epoch = len(loader)
        except TypeError:
            updates_per_epoch = 0
        self.planned_task_optimizer_steps = (
            self.epochs_per_task * updates_per_epoch if updates_per_epoch else None
        )
        self._reset_schedule_for_task(updates_per_epoch=updates_per_epoch)
        epoch_progress = self._progress(
            range(self.epochs_per_task),
            desc=f"CLAD-D domain {task_id + 1}/{self.benchmark.num_tasks}",
            unit="epoch",
            leave=True,
        )
        last_action_name: Optional[str] = None
        for event in reversed(self.action_events):
            if event.applies_to_task_id == task_id:
                last_action_name = event.action_name
                break

        epoch_progress.set_postfix({"phase": "validation"}, refresh=True)
        initial = self._incoming_domain_state(task_id)
        self.at_domain_boundary = True
        self._decision_boundary(
            task_id,
            initial,
            phase="task_start",
            terminal=False,
            applies_to_task_id=task_id,
        )
        if self.action_events:
            last_action_name = self.action_events[-1].action_name
        epoch_progress.set_postfix(
            {
                "phase": "train",
                "mAP50": f"{initial.equal_domain_map50:.4f}",
                "action": last_action_name or "none",
            },
            refresh=True,
        )
        self.at_domain_boundary = False

        epoch_reports = []
        final: Optional[DomainAP50Summary] = None
        for epoch in epoch_progress:
            report = self._train_epoch(loader, epoch=epoch + 1)
            self.scheduler.step()
            self.current_epoch = epoch + 1
            epoch_reports.append({"epoch": self.current_epoch, **report})

            epoch_postfix = {
                "phase": "train",
                "loss": f"{report['current_detection_loss']:.4f}",
                "replay": f"{report['replay_detection_loss']:.4f}",
                "kd": f"{report['replay_distillation_loss']:.4f}",
                "lr": f"{self._display_learning_rate():.2e}",
                "buffer": (
                    f"{len(self.replay_buffer)}/{self.replay_buffer.current_capacity}"
                ),
                "replay_w": f"{self.replay_loss_weight:g}",
                "step": self.optimizer_step,
            }
            if last_action_name is not None:
                epoch_postfix["action"] = last_action_name

            if self.current_epoch % self.decision_frequency_epochs:
                epoch_progress.set_postfix(epoch_postfix, refresh=True)
                continue
            epoch_postfix["phase"] = "validation"
            epoch_progress.set_postfix(epoch_postfix, refresh=True)
            boundary_result = self.evaluate_seen_domains(task_id)
            # A freeze action lasts exactly the interval that just completed.
            # Release it before constructing the successor action mask.
            self.model.unfreeze_group("backbone")
            task_end = self.current_epoch == self.epochs_per_task
            terminal = task_end and task_id == self.benchmark.num_tasks - 1
            self.at_domain_boundary = task_end and not terminal
            applies_to_task_id = task_id + 1 if task_end and not terminal else task_id
            self._decision_boundary(
                task_id,
                boundary_result,
                phase="task_end" if task_end else "interval",
                terminal=terminal,
                applies_to_task_id=None if terminal else applies_to_task_id,
                defer_successor=task_end and not terminal,
            )
            if not task_end and self.action_events:
                last_action_name = self.action_events[-1].action_name
            epoch_postfix.update(
                phase="train" if not task_end else "complete",
                mAP50=f"{boundary_result.equal_domain_map50:.4f}",
                action=last_action_name or "none",
            )
            epoch_progress.set_postfix(epoch_postfix, refresh=True)
            self.at_domain_boundary = False
            if task_end:
                final = boundary_result

        if final is None:
            raise RuntimeError("The final epoch did not reach a decision boundary")
        is_final_task = task_id == self.benchmark.num_tasks - 1
        if is_final_task and (
            self.pending_decision is not None
            or self.pending_outcome_metrics is not None
            or self.pending_old_domain_results is not None
        ):
            raise RuntimeError("Final transition failed to terminate the DQN episode")
        if not is_final_task and not self.fixed_schedule and (
            self.pending_decision is None
            or self.pending_outcome_metrics is None
            or self.pending_old_domain_results is None
        ):
            raise RuntimeError(
                "Domain boundary did not defer the final interval transition"
            )

        self.metric_history.record(task_id, final.per_domain)
        admission = self._admit_completed_domain(task_id)
        retained = int(admission["retained_from_task"])
        if not is_final_task:
            teacher_cache_size = self._build_distillation_target_cache(
                teacher_task_id=task_id
            )
        else:
            self._distillation_target_cache.clear()
            self._flipped_teacher_boxes.clear()
            teacher_cache_size = 0
        check_model_health(self.model)

        self.next_task_id = task_id + 1
        self.planned_task_optimizer_steps = None
        mean_distillation_loss = sum(
            float(epoch_report.get("replay_distillation_loss", 0.0))
            for epoch_report in epoch_reports
        ) / max(1, len(epoch_reports))
        mean_distillation_matches = sum(
            float(epoch_report.get("distillation_matches_per_batch", 0.0))
            for epoch_report in epoch_reports
        ) / max(1, len(epoch_reports))
        report = {
            "task_id": task_id,
            "controller_probe": final.as_dict(),
            "controller_feedback": deepcopy(
                self.evaluation_history[-1]["controller_metrics"]
            ),
            "epochs": epoch_reports,
            "replay_images_available_during_task": (
                replay_images_available_during_task
            ),
            "replay_domain_counts_during_task": (replay_domain_counts_during_task),
            "replay_images_after_admission": int(admission["size_after"]),
            "replay_domain_counts_after_admission": deepcopy(
                admission["domain_counts_after"]
            ),
            "replay_class_image_counts_after_admission": deepcopy(
                admission.get("class_image_counts_after", {})
            ),
            "replay_class_object_counts_after_admission": deepcopy(
                admission.get("class_object_counts_after", {})
            ),
            "replay_object_size_counts_after_admission": deepcopy(
                admission.get("object_size_counts_after", {})
            ),
            "admitted_candidates_from_task": int(admission["considered"]),
            "replay_loss_weight_start": replay_loss_weight_start,
            "replay_loss_weight_end": float(self.replay_loss_weight),
            "distillation_teacher_targets": int(teacher_cache_size),
            "mean_replay_distillation_loss": mean_distillation_loss,
            "mean_distillation_matches_per_batch": mean_distillation_matches,
            # Backward-compatible aliases are explicitly post-admission.
            "replay_images": len(self.replay_buffer),
            "replay_capacity": self.replay_buffer.current_capacity,
            "retained_from_task": retained,
            "model_parameters": self.model.get_num_parameters(),
        }
        self.task_reports.append(report)

        if bool(self.config.get("checkpointing", {}).get("save_every_task", True)):
            directory = str(
                self.config.get("logging", {}).get("save_dir", "./checkpoints/carl_d")
            )
            self.save_checkpoint(
                os.path.join(
                    directory,
                    f"task_{task_id + 1}_{self.run_timestamp}.pt",
                )
            )
        return report

    def train(self) -> Dict[str, Any]:
        for task_id in range(self.next_task_id, self.benchmark.num_tasks):
            self.train_task(task_id)
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        history = self.metric_history.summary()
        return {
            "benchmark": "CLAD-D",
            "completed_tasks": self.next_task_id,
            "optimizer_steps": self.optimizer_step,
            "controller_probe": history,
            "task_reports": deepcopy(self.task_reports),
            "action_distribution": self.controller.get_action_distribution(),
            "mean_reward_by_action": (self.controller.get_mean_reward_by_action()),
            "controller_action_diagnostics": (self.controller.get_action_diagnostics()),
            "actions": [event.to_dict() for event in self.action_events],
            "replay_images": len(self.replay_buffer),
            "replay_capacity": self.replay_buffer.current_capacity,
            "replay_domain_counts": self.replay_buffer.domain_counts(),
            "replay_class_image_counts": self.replay_buffer.class_image_counts(),
            "replay_class_object_counts": self.replay_buffer.class_object_counts(),
            "replay_object_size_counts": self.replay_buffer.object_size_counts(),
            "replay_loss_weight": float(self.replay_loss_weight),
            "replay_loss_weight_levels": list(self.replay_loss_weight_levels),
            "stagewise_protection": bool(self.stagewise_protection),
            "resnet_stage_lr_multipliers_after_task1": list(
                self.resnet_stage_lr_multipliers
            ),
            "distillation_enabled": bool(self.distillation_enabled),
            "distillation_loss_weight": float(self.distillation_loss_weight),
            "distillation_temperature": float(self.distillation_temperature),
            "model_parameters": self.model.get_num_parameters(),
            "base_model_parameters": self.base_parameter_count,
            "max_model_parameters": self.max_parameter_count,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "adaptation_semantics_version": 5,
            "freeze_available_at_epoch": self.freeze_available_at_epoch,
            "replay_weight_available_at_epoch": self.replay_weight_available_at_epoch,
            "feedback_annotation_counts": deepcopy(self._feedback_annotation_counts),
            "next_task_id": self.next_task_id,
            "current_task_id": self.current_task_id,
            "current_epoch": self.current_epoch,
            "optimizer_step": self.optimizer_step,
            "task_optimizer_step": self.task_optimizer_step,
            "warmup_optimizer_step": self._warmup_optimizer_step,
            "warmup_total_optimizer_steps": self._warmup_total_optimizer_steps,
            "at_domain_boundary": self.at_domain_boundary,
            "loss_ema": self.loss_ema,
            "recognition_loss_ema": self.recognition_loss_ema,
            "localization_loss_ema": self.localization_loss_ema,
            "loss_ema_updates": self.loss_ema_updates,
            "latest_rpn_recall": self.latest_rpn_recall,
            "replay_loss_weight": self.replay_loss_weight,
            "base_parameter_count": self.base_parameter_count,
            "max_parameter_count": self.max_parameter_count,
            "resume_config_fingerprint": self.resume_config_fingerprint,
            "best_domain_ap50": deepcopy(self.best_domain_ap50),
            "pending_decision": deepcopy(self.pending_decision),
            "pending_outcome_metrics": deepcopy(self.pending_outcome_metrics),
            "pending_old_domain_results": deepcopy(self.pending_old_domain_results),
            "last_structural_expansion_decision_step": (
                self.last_structural_expansion_decision_step
            ),
            "refinement_cursor": int(self._refinement_cursor),
            "task_class_count_cache": deepcopy(self._task_class_count_cache),
            "distillation_target_cache": {
                key: {
                    "boxes": value["boxes"].detach().cpu().clone(),
                    "logits": value["logits"].detach().cpu().clone(),
                }
                for key, value in self._distillation_target_cache.items()
            },
            "metric_history": self.metric_history.state_dict(),
            "evaluation_history": deepcopy(self.evaluation_history),
            "action_events": [event.to_dict() for event in self.action_events],
            "task_reports": deepcopy(self.task_reports),
        }

    def _load_trainer_state(self, state: Mapping[str, Any]) -> None:
        if state.get("adaptation_semantics_version") != 5:
            raise ValueError(
                "This checkpoint uses historical RL feedback/freezing semantics. "
                "Start a fresh training run; evaluation-only loading remains supported."
            )
        if state.get("resume_config_fingerprint") != self.resume_config_fingerprint:
            raise ValueError("Resume-critical CARL-D configuration differs")
        self.freeze_available_at_epoch = int(state["freeze_available_at_epoch"])
        self.replay_weight_available_at_epoch = int(
            state["replay_weight_available_at_epoch"]
        )
        self._feedback_annotation_counts = {
            int(domain): {int(class_id): int(count) for class_id, count in counts.items()}
            for domain, counts in state.get("feedback_annotation_counts", {}).items()
        }
        if int(state["base_parameter_count"]) != self.base_parameter_count:
            raise ValueError("Checkpoint base detector parameter count differs")
        if int(state["max_parameter_count"]) != self.max_parameter_count:
            raise ValueError("Checkpoint detector parameter budget differs")
        self.next_task_id = int(state["next_task_id"])
        self.current_task_id = int(state["current_task_id"])
        self.current_epoch = int(state["current_epoch"])
        self.optimizer_step = int(state["optimizer_step"])
        self.task_optimizer_step = int(state["task_optimizer_step"])
        self._warmup_optimizer_step = int(state.get("warmup_optimizer_step", 0))
        self._warmup_total_optimizer_steps = int(
            state.get("warmup_total_optimizer_steps", 0)
        )
        self.at_domain_boundary = bool(state["at_domain_boundary"])
        self.loss_ema = float(state["loss_ema"])
        self.recognition_loss_ema = float(state["recognition_loss_ema"])
        self.localization_loss_ema = float(state["localization_loss_ema"])
        self.loss_ema_updates = int(state["loss_ema_updates"])
        self.latest_rpn_recall = float(state["latest_rpn_recall"])
        replay_loss_weight = float(state["replay_loss_weight"])
        if not any(
            math.isclose(replay_loss_weight, value)
            for value in self.replay_loss_weight_levels
        ):
            raise ValueError("Checkpoint replay loss weight is not a configured level")
        self.replay_loss_weight = replay_loss_weight
        self.best_domain_ap50 = {
            int(key): float(value) for key, value in state["best_domain_ap50"].items()
        }
        self.pending_decision = deepcopy(state["pending_decision"])
        self.pending_outcome_metrics = deepcopy(state.get("pending_outcome_metrics"))
        raw_old_results = state.get("pending_old_domain_results")
        self.pending_old_domain_results = (
            None
            if raw_old_results is None
            else {
                int(domain): deepcopy(dict(result))
                for domain, result in raw_old_results.items()
            }
        )
        raw_expansion_step = state.get("last_structural_expansion_decision_step")
        self.last_structural_expansion_decision_step = (
            None if raw_expansion_step is None else int(raw_expansion_step)
        )
        self._refinement_cursor = int(state.get("refinement_cursor", 0))
        self._task_class_count_cache = {
            int(task): {int(class_id): int(count) for class_id, count in counts.items()}
            for task, counts in state.get("task_class_count_cache", {}).items()
        }
        self._distillation_target_cache = {}
        self._flipped_teacher_boxes = {}
        for key, value in state.get("distillation_target_cache", {}).items():
            boxes = value["boxes"].detach().cpu().to(dtype=torch.float32)
            logits = (
                value["logits"].detach().cpu().to(dtype=self.distillation_cache_dtype)
            )
            if self.device.type == "cuda":
                boxes = boxes.pin_memory()
                logits = logits.pin_memory()
            self._distillation_target_cache[(int(key[0]), self._image_id(key[1]))] = {
                "boxes": boxes,
                "logits": logits,
            }
        if (
            self.distillation_enabled
            and self.next_task_id in {1, 2, 3}
            and len(self._distillation_target_cache) != len(self.replay_buffer)
        ):
            raise ValueError(
                "Checkpoint does not contain complete replay distillation targets"
            )
        self.metric_history.load_state_dict(state["metric_history"])
        self.evaluation_history = deepcopy(list(state["evaluation_history"]))
        self.action_events = [
            ActionExecutionEvent.from_mapping(event) for event in state["action_events"]
        ]
        self.task_reports = deepcopy(list(state["task_reports"]))

    def save_checkpoint(self, path: str) -> None:
        if self.planned_task_optimizer_steps is not None:
            raise RuntimeError(
                "Exact CARL-D checkpoints can only be saved at a task boundary"
            )
        save_carl_d_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            replay_buffer=self.replay_buffer,
            controller=self.controller,
            trainer_state=self.state_dict(),
            config=self.config,
            dataset_fingerprint=self.dataset_fingerprint,
        )

    def load_checkpoint(
        self, path: str, *, load_training_state: bool = True
    ) -> Dict[str, Any]:
        # Recreate dynamic adapters before constructing an optimizer whose
        # parameter groups must exactly match the checkpoint.
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if load_training_state and (
            raw.get("trainer_state", {}).get("adaptation_semantics_version") != 5
        ):
            raise ValueError(
                "Cannot resume historical RL feedback/freezing semantics. "
                "Start a fresh run or load with load_training_state=False for evaluation."
            )
        if raw.get("checkpoint_type") != CHECKPOINT_TYPE:
            raise ValueError("Checkpoint belongs to an incompatible framework")
        if int(raw.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported CARL-D checkpoint version: "
                f"{raw.get('checkpoint_version')}"
            )
        if str(raw.get("dataset_fingerprint", "")) != self.dataset_fingerprint:
            raise ValueError("Dataset split fingerprint differs from checkpoint")
        checkpoint_config = raw.get("config")
        if not isinstance(checkpoint_config, Mapping):
            raise ValueError("Checkpoint does not contain a CARL-D configuration")
        if load_training_state and (
            resume_config_fingerprint(checkpoint_config)
            != self.resume_config_fingerprint
        ):
            raise ValueError("Resume-critical CARL-D configuration differs")
        self.model.apply_architecture_manifest(raw["architecture_manifest"])

        if load_training_state:
            self.optimizer = self._new_optimizer()
            self.scheduler = self._new_scheduler()
        result = restore_carl_d_checkpoint(
            raw,
            model=self.model,
            optimizer=self.optimizer if load_training_state else None,
            scheduler=self.scheduler if load_training_state else None,
            scaler=self.scaler if load_training_state else None,
            replay_buffer=self.replay_buffer if load_training_state else None,
            controller=self.controller if load_training_state else None,
            expected_dataset_fingerprint=self.dataset_fingerprint,
            restore_rng=load_training_state,
        )
        if load_training_state:
            self._load_trainer_state(result["trainer_state"])
            self._synchronize_replay_tensor_cache()
        return result


__all__ = [
    "ActionExecutionEvent",
    "CARLDTrainer",
    "check_model_health",
    "dataset_split_fingerprint",
    "resume_config_fingerprint",
    "validate_carl_d_config",
]
