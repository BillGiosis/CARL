"""Shared ResNet50 detector, adapters, replay distillation and checkpoint contract."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .adapters import (
    ExpandableBackboneWithFPN,
    ExpandableRoIBoxHead,
)
from .roi_heads import ClassBalancedRoIHeads

CARL_D_NUM_CLASSES = 7  # background 0 plus the six CLAD-D foreground classes
EXPANDABLE_GROUPS = ("backbone", "fpn", "roi")
RESNET50_ARCHITECTURE = "fasterrcnn_resnet50_fpn_v2"
SUPPORTED_ARCHITECTURES = (RESNET50_ARCHITECTURE,)


def _first_conv_in_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"could not infer input channels from {type(module).__name__}")


def _resolve_weights(weights: Any):
    if weights is None or weights is False:
        return None
    if isinstance(weights, FasterRCNN_ResNet50_FPN_V2_Weights):
        return weights
    if weights is True:
        return FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    if isinstance(weights, str):
        value = weights.upper()
        if value in {"NONE", "RANDOM"}:
            return None
        if value in {"DEFAULT", "COCO_V1"}:
            return FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    raise ValueError("weights must be None, DEFAULT, COCO_V1, or a V2 weights enum")


def _module_device_and_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return torch.device("cpu"), torch.get_default_dtype()
    return parameter.device, parameter.dtype


class CARLDDetector(nn.Module):
    """A shared-head CLAD-D detector with dynamically expandable adapters.

    ``weights`` defaults to the complete COCO-pretrained V2 detector. Pass ``weights=None`` for deliberately
    random-initialized runs and to prevent a download.
    """

    def __init__(
        self,
        *,
        architecture: str = RESNET50_ARCHITECTURE,
        num_classes: int = CARL_D_NUM_CLASSES,
        weights: Any = "DEFAULT",
        adapter_reduction: int = 16,
        adapter_merge: str = "parallel",
        class_balanced_roi_loss: bool = True,
        preserve_coco_predictor_rows: bool = True,
        detector: nn.Module | None = None,
        **detector_kwargs: Any,
    ) -> None:
        super().__init__()
        if num_classes != CARL_D_NUM_CLASSES:
            raise ValueError(
                "CARL-D requires 7 detector outputs: background 0 and foreground 1..6"
            )
        if adapter_reduction <= 0:
            raise ValueError("adapter_reduction must be positive")
        if not isinstance(preserve_coco_predictor_rows, bool):
            raise TypeError("preserve_coco_predictor_rows must be boolean")
        self.num_classes = int(num_classes)
        self.architecture = str(architecture).lower()
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {SUPPORTED_ARCHITECTURES}, "
                f"got {architecture!r}"
            )
        self.adapter_reduction = int(adapter_reduction)
        self.adapter_merge = str(adapter_merge).lower()
        if self.adapter_merge != "parallel":
            raise ValueError("CARL supports only parallel adapter composition")
        if not isinstance(class_balanced_roi_loss, bool):
            raise TypeError("class_balanced_roi_loss must be boolean")
        self.class_balanced_roi_loss = class_balanced_roi_loss
        self.preserve_coco_predictor_rows = bool(preserve_coco_predictor_rows)

        if detector is None:
            build_kwargs = dict(detector_kwargs)
            build_kwargs.pop("num_classes", None)
            resolved_weights = _resolve_weights(weights)
            if resolved_weights is None:
                build_kwargs["num_classes"] = self.num_classes
            detector = fasterrcnn_resnet50_fpn_v2(
                weights=resolved_weights,
                weights_backbone=None,
                **build_kwargs,
            )
        elif detector_kwargs:
            raise ValueError("detector_kwargs cannot be used with an injected detector")

        self.detector = detector
        self._install_clad_d_predictor()
        self._install_expandable_modules()
        if self.class_balanced_roi_loss:
            self._install_class_balanced_roi_heads()

        self._freeze_snapshots: dict[str, list[tuple[nn.Parameter, bool]]] = {}
        self._freeze_preserve_adapters: dict[str, bool] = {}
        self._frozen_until: dict[str, int | None] = {}
        self._frozen_module_modes: dict[str, list[tuple[nn.Module, bool]]] = {}

    def _install_clad_d_predictor(self) -> None:
        predictor = self.detector.roi_heads.box_predictor
        self.detector.roi_heads.box_predictor = self._clad_d_predictor_from_coco(
            predictor
        )

    def _clad_d_predictor_from_coco(self, predictor: nn.Module) -> nn.Module:
        if not hasattr(predictor, "cls_score") or not hasattr(
            predictor.cls_score, "in_features"
        ):
            raise TypeError("detector must expose a Fast R-CNN box predictor")
        if int(predictor.cls_score.out_features) == self.num_classes:
            return predictor

        device, dtype = _module_device_and_dtype(predictor)
        replacement = FastRCNNPredictor(
            int(predictor.cls_score.in_features), self.num_classes
        ).to(device=device, dtype=dtype)
        # Preserve only exact COCO -> CLAD-D semantic matches.  Torchvision's
        # COCO predictor uses background=0, person=1, car=3, and truck=8;
        # CLAD-D uses background=0, pedestrian=1, car=3, and truck=4.
        # The remaining detector categories keep their normal random
        # initialization because their mappings are ambiguous.
        exact_rows = {0: 0, 1: 1, 3: 3, 8: 4}
        with torch.no_grad():
            for coco_id, cladd_id in exact_rows.items():
                if not self.preserve_coco_predictor_rows:
                    break
                if (
                    coco_id >= predictor.cls_score.out_features
                    or cladd_id >= replacement.cls_score.out_features
                ):
                    continue
                replacement.cls_score.weight[cladd_id].copy_(
                    predictor.cls_score.weight[coco_id]
                )
                replacement.cls_score.bias[cladd_id].copy_(
                    predictor.cls_score.bias[coco_id]
                )

                source_start = 4 * coco_id
                target_start = 4 * cladd_id
                if (
                    source_start + 4 <= predictor.bbox_pred.out_features
                    and target_start + 4 <= replacement.bbox_pred.out_features
                ):
                    replacement.bbox_pred.weight[target_start : target_start + 4].copy_(
                        predictor.bbox_pred.weight[source_start : source_start + 4]
                    )
                    replacement.bbox_pred.bias[target_start : target_start + 4].copy_(
                        predictor.bbox_pred.bias[source_start : source_start + 4]
                    )
        return replacement

    def _install_expandable_modules(self) -> None:
        backbone = self.detector.backbone
        if not isinstance(backbone, ExpandableBackboneWithFPN):
            explicit_names = getattr(backbone, "carl_feature_names", None)
            explicit_channels = getattr(backbone, "carl_feature_channels", None)
            explicit_fpn_names = getattr(backbone, "carl_fpn_feature_names", None)
            if explicit_names is not None or explicit_channels is not None:
                if explicit_names is None or explicit_channels is None:
                    raise TypeError(
                        "detector backbone must provide both CARL feature names and channels"
                    )
                feature_names = tuple(str(name) for name in explicit_names)
                backbone_channels = tuple(int(width) for width in explicit_channels)
                if len(feature_names) != len(backbone_channels):
                    raise ValueError(
                        "explicit backbone feature metadata is inconsistent"
                    )
            else:
                if not hasattr(backbone.body, "return_layers"):
                    raise TypeError(
                        "detector backbone must expose return_layers or explicit CARL metadata"
                    )
                feature_names = tuple(
                    str(name) for name in backbone.body.return_layers.values()
                )
                inner_blocks = tuple(backbone.fpn.inner_blocks)
                if len(feature_names) != len(inner_blocks):
                    raise ValueError("FPN inputs do not match backbone return layers")
                backbone_channels = tuple(
                    _first_conv_in_channels(block) for block in inner_blocks
                )

            fpn_feature_names = (
                list(str(name) for name in explicit_fpn_names)
                if explicit_fpn_names is not None
                else list(feature_names)
            )
            if isinstance(backbone.fpn.extra_blocks, LastLevelMaxPool):
                if "pool" not in fpn_feature_names:
                    fpn_feature_names.append("pool")
            else:
                raise TypeError(
                    "CARL-D requires a two-stage detector FPN with LastLevelMaxPool"
                )
            self.detector.backbone = ExpandableBackboneWithFPN(
                backbone,
                feature_names,
                backbone_channels,
                fpn_feature_names,
                adapter_merge=self.adapter_merge,
            )
        else:
            backbone.set_adapter_merge(self.adapter_merge)

        box_head = self.detector.roi_heads.box_head
        if not isinstance(box_head, ExpandableRoIBoxHead):
            output_features = int(
                self.detector.roi_heads.box_predictor.cls_score.in_features
            )
            self.detector.roi_heads.box_head = ExpandableRoIBoxHead(
                box_head, output_features, adapter_merge=self.adapter_merge
            )
        else:
            box_head.set_adapter_merge(self.adapter_merge)

    def _install_class_balanced_roi_heads(self) -> None:
        roi_heads = self.detector.roi_heads
        if isinstance(roi_heads, ClassBalancedRoIHeads):
            return
        self.detector.roi_heads = ClassBalancedRoIHeads.from_roi_heads(
            roi_heads, self.num_classes
        )

    def set_roi_class_weights(self, weights: torch.Tensor) -> None:
        """Install normalized foreground weights; background must stay one."""
        roi_heads = self.detector.roi_heads
        if not isinstance(roi_heads, ClassBalancedRoIHeads):
            if not torch.allclose(
                torch.as_tensor(weights, dtype=torch.float32),
                torch.ones(self.num_classes, dtype=torch.float32),
            ):
                raise RuntimeError("class-balanced RoI loss is disabled")
            return
        roi_heads.set_positive_class_weights(weights)

    def adapter_generation_count(self, group: str) -> int:
        if group == "backbone":
            return len(self.backbone.backbone_adapters)
        if group == "fpn":
            return len(self.backbone.fpn_adapters)
        if group == "roi":
            return len(self.detector.roi_heads.box_head.adapters)
        raise ValueError(f"group must be one of {EXPANDABLE_GROUPS}, got {group!r}")

    @property
    def backbone(self) -> ExpandableBackboneWithFPN:
        return self.detector.backbone

    def forward(
        self,
        images: list[torch.Tensor] | tuple[torch.Tensor, ...],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ):
        return self.detector(images, targets)

    @torch.no_grad()
    def extract_replay_distillation_targets(
        self,
        images: list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        max_proposals_per_image: int,
    ) -> list[dict[str, torch.Tensor]]:
        """Extract compact final-head teacher targets for replay images."""
        if not images:
            raise ValueError("distillation-target images cannot be empty")

        was_training = self.training
        self.eval()
        try:
            transformed_images, _ = self.detector.transform(list(images), None)
            features = self.detector.backbone(transformed_images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict((("0", features),))
            proposals, _ = self.detector.rpn(transformed_images, features, None)
            roi_heads = self.detector.roi_heads
            if max_proposals_per_image <= 0:
                raise ValueError("max_proposals_per_image must be positive")
            pooled = roi_heads.box_roi_pool(
                features, proposals, transformed_images.image_sizes
            )
            logits, _ = roi_heads.box_predictor(roi_heads.box_head(pooled))
            result = []
            for boxes, values, (height, width) in zip(
                proposals,
                logits.float().split([len(p) for p in proposals]),
                transformed_images.image_sizes,
            ):
                # Select half by foreground confidence and half by entropy.
                count = min(max_proposals_per_image, len(boxes))
                probabilities = values.softmax(-1)
                entropy = -(
                    probabilities
                    * probabilities.clamp_min(
                        torch.finfo(probabilities.dtype).tiny
                    ).log()
                ).sum(-1)
                selected = torch.zeros(
                    len(boxes), dtype=torch.bool, device=boxes.device
                )
                foreground_count = (count + 1) // 2
                selected[(1 - probabilities[:, 0]).topk(foreground_count).indices] = (
                    True
                )
                remaining = count - foreground_count
                if remaining:
                    selected[
                        entropy.masked_fill(selected, -float("inf"))
                        .topk(remaining)
                        .indices
                    ] = True
                scale = boxes.new_tensor([width, height, width, height])
                result.append(
                    {
                        "boxes": (boxes[selected] / scale).clamp(0, 1).float(),
                        "logits": values[selected],
                    }
                )
            return result
        finally:
            self.train(was_training)

    def forward_with_replay_distillation(
        self,
        images: list[torch.Tensor] | tuple[torch.Tensor, ...],
        targets: list[dict[str, torch.Tensor]],
        teacher_targets: list[Mapping[str, torch.Tensor]],
        *,
        temperature: float,
        match_iou_threshold: float,
    ) -> tuple[Mapping[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return native replay losses plus matched final-head logit KD.

        The normal replay detector forward captures its already-computed
        final-head sampled proposals and logits (single head for R50). Matching those proposals to
        cached teacher proposals avoids a second student backbone/head pass.
        """
        if len(images) != len(targets) or len(images) != len(teacher_targets):
            raise ValueError("replay images, targets, and teacher targets must align")
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise ValueError("distillation temperature must be finite and positive")
        if not 0.0 <= float(match_iou_threshold) <= 1.0:
            raise ValueError("distillation match IoU must be in [0, 1]")

        roi_heads = self.detector.roi_heads
        captured = {}

        def capture_boxes(module, args):
            captured["boxes"] = [p.detach() for p in args[1]]
            captured["shapes"] = args[2]

        def capture_logits(module, args, output):
            captured["logits"] = output[0]

        handles = [
            roi_heads.box_roi_pool.register_forward_pre_hook(capture_boxes),
            roi_heads.box_predictor.register_forward_hook(capture_logits),
        ]
        try:
            loss_dict = self.detector(list(images), list(targets))
        finally:
            for handle in handles:
                handle.remove()
        student_boxes = captured["boxes"]
        image_shapes = captured["shapes"]
        student_logits = captured["logits"].split([len(p) for p in student_boxes])
        temperature = float(temperature)
        threshold = float(match_iou_threshold)
        kd_sum = student_logits[0].sum() * 0.0
        matched_count = kd_sum.detach().clone()
        for proposals, logits, image_shape, teacher in zip(
            student_boxes, student_logits, image_shapes, teacher_targets
        ):
            teacher_boxes = torch.as_tensor(
                teacher["boxes"], device=logits.device, dtype=torch.float32
            )
            teacher_logits = torch.as_tensor(
                teacher["logits"], device=logits.device, dtype=torch.float32
            )
            if (
                teacher_boxes.ndim != 2
                or teacher_boxes.shape[-1] != 4
                or teacher_logits.ndim != 2
                or teacher_logits.shape[0] != teacher_boxes.shape[0]
                or teacher_logits.shape[1] != self.num_classes
            ):
                raise ValueError("cached distillation targets have invalid shapes")
            if proposals.numel() == 0 or teacher_boxes.numel() == 0:
                continue
            height, width = image_shape
            scale = proposals.new_tensor(
                [float(width), float(height), float(width), float(height)]
            )
            normalized_student_boxes = (proposals / scale).clamp(0.0, 1.0)
            best_iou, best_teacher = box_iou(
                normalized_student_boxes.float(), teacher_boxes
            ).max(dim=1)
            selected_teacher = teacher_logits[best_teacher]
            per_proposal = F.kl_div(
                F.log_softmax(logits.float() / temperature, dim=-1),
                F.softmax(selected_teacher / temperature, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            match_weights = (best_iou >= threshold).to(per_proposal.dtype)
            kd_sum = kd_sum + (per_proposal * match_weights).sum() * (
                temperature * temperature
            )
            matched_count = matched_count + match_weights.sum()

        kd_loss = kd_sum / matched_count.clamp_min(1.0)
        return loss_dict, kd_loss, matched_count

    @torch.no_grad()
    def predict_with_rpn_recall(
        self,
        images: list[torch.Tensor] | tuple[torch.Tensor, ...],
        targets: list[dict[str, torch.Tensor]] | tuple[dict[str, torch.Tensor], ...],
        iou_threshold: float = 0.5,
        *,
        return_tensor_counts: bool = False,
    ) -> tuple[list[dict[str, torch.Tensor]], int | torch.Tensor, int]:
        """Return detections and exact post-NMS RPN proposal recall counts.

        A ground-truth box is matched when at least one of the proposals passed
        from the RPN to the RoI heads has IoU greater than or equal to
        ``iou_threshold``. Every supplied ground-truth box is counted once in
        ``total_gt``; a proposal may recall more than one ground-truth box.

        This method executes the Torchvision Generalized R-CNN stages directly,
        so the returned detections and proposal counts come from one shared
        backbone/RPN pass. It temporarily uses inference mode and restores the
        caller's train/eval mode before returning.
        ``return_tensor_counts`` keeps the matched count on-device so callers
        can aggregate it without synchronizing once per batch.
        """
        if not 0.0 <= float(iou_threshold) <= 1.0:
            raise ValueError("iou_threshold must be in the 0..1 range")
        if not images:
            raise ValueError("images cannot be empty")
        if len(images) != len(targets):
            raise ValueError("images and targets must contain the same samples")

        original_image_sizes: list[tuple[int, int]] = []
        prepared_targets: list[dict[str, torch.Tensor]] = []
        for index, (image, target) in enumerate(zip(images, targets)):
            if not torch.is_tensor(image) or image.ndim != 3:
                raise ValueError(f"image {index} must be a [C, H, W] tensor")
            if "boxes" not in target or not torch.is_tensor(target["boxes"]):
                raise ValueError(f"target {index} must contain tensor boxes")
            boxes = target["boxes"]
            if boxes.ndim != 2 or boxes.shape[-1] != 4:
                raise ValueError(f"target {index} boxes must have shape [N, 4]")
            original_image_sizes.append((int(image.shape[-2]), int(image.shape[-1])))
            prepared_targets.append(
                {
                    key: value.to(device=image.device, non_blocking=True)
                    if torch.is_tensor(value)
                    else value
                    for key, value in target.items()
                }
            )

        was_training = self.training
        self.eval()
        try:
            transformed_images, transformed_targets = self.detector.transform(
                list(images), prepared_targets
            )
            if transformed_targets is None:  # defensive: targets were supplied
                raise RuntimeError(
                    "detector transform discarded proposal-recall targets"
                )

            features = self.detector.backbone(transformed_images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict((("0", features),))
            proposals, _ = self.detector.rpn(transformed_images, features, None)
            detections, _ = self.detector.roi_heads(
                features,
                proposals,
                transformed_images.image_sizes,
                None,
            )
            detections = self.detector.transform.postprocess(
                detections,
                transformed_images.image_sizes,
                original_image_sizes,
            )

            matched_gt = torch.zeros(
                (), dtype=torch.int64, device=transformed_images.tensors.device
            )
            total_gt = 0
            threshold = float(iou_threshold)
            for proposal_boxes, target in zip(proposals, transformed_targets):
                ground_truth = target["boxes"]
                total_gt += int(ground_truth.shape[0])
                if ground_truth.numel() == 0 or proposal_boxes.numel() == 0:
                    continue
                best_iou = box_iou(ground_truth, proposal_boxes).max(dim=1).values
                matched_gt += (best_iou >= threshold).sum()
            # Accumulate integer counts without a per-batch synchronization.
            matched = matched_gt if return_tensor_counts else int(matched_gt.item())
            return detections, matched, total_gt
        finally:
            self.train(was_training)

    @torch.no_grad()
    def predict_with_refinement_signals(
        self,
        images: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> tuple[
        list[dict[str, torch.Tensor]],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Return detections, pre-NMS RoI probabilities, and image features.

        The probability tensors contain one row per RPN proposal and all seven
        detector probabilities before score thresholding and NMS.  The feature
        vectors concatenate global averages from the immutable FPN outputs and
        are L2-normalized for replay-redundancy comparisons.
        """

        if not images:
            raise ValueError("images cannot be empty")
        original_image_sizes: list[tuple[int, int]] = []
        for index, image in enumerate(images):
            if not torch.is_tensor(image) or image.ndim != 3:
                raise ValueError(f"image {index} must be a [C, H, W] tensor")
            original_image_sizes.append((int(image.shape[-2]), int(image.shape[-1])))

        was_training = self.training
        self.eval()
        try:
            transformed_images, _ = self.detector.transform(list(images), None)
            features = self.detector.backbone(transformed_images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict((("0", features),))
            proposals, _ = self.detector.rpn(transformed_images, features, None)

            roi_heads = self.detector.roi_heads
            box_features = roi_heads.box_roi_pool(
                features, proposals, transformed_images.image_sizes
            )
            box_features = roi_heads.box_head(box_features)
            class_logits, box_regression = roi_heads.box_predictor(box_features)
            boxes, scores, labels = roi_heads.postprocess_detections(
                class_logits,
                box_regression,
                proposals,
                transformed_images.image_sizes,
            )
            detections = [
                {"boxes": box, "scores": score, "labels": label}
                for box, score, label in zip(boxes, scores, labels)
            ]
            probabilities = list(
                class_logits.float()
                .softmax(dim=-1)
                .split([len(proposal) for proposal in proposals])
            )
            detections = self.detector.transform.postprocess(
                detections,
                transformed_images.image_sizes,
                original_image_sizes,
            )

            feature_maps = tuple(features.values())
            embeddings: list[torch.Tensor] = []
            for image_index in range(len(images)):
                pooled = torch.cat(
                    [
                        feature[image_index].float().mean(dim=(-2, -1))
                        for feature in feature_maps
                    ]
                )
                embeddings.append(pooled / pooled.norm(p=2).clamp_min(1e-12))
            return detections, list(probabilities), embeddings
        finally:
            self.train(was_training)

    def estimate_expansion_params(
        self, group: str, reduction: int | None = None
    ) -> int:
        """Return the exact parameter delta for the next expansion generation."""
        reduction = self.adapter_reduction if reduction is None else int(reduction)
        if group == "backbone" or group == "fpn":
            return self.backbone.estimate_expansion_params(group, reduction)
        if group == "roi":
            return self.detector.roi_heads.box_head.estimate_expansion_params(reduction)
        raise ValueError(
            f"expandable group must be one of {EXPANDABLE_GROUPS}, got {group!r}"
        )

    def expand_group(self, group: str, reduction: int | None = None) -> int:
        """Add one zero-initialized adapter generation and return its exact size."""
        reduction = self.adapter_reduction if reduction is None else int(reduction)
        expected = self.estimate_expansion_params(group, reduction)
        before = self.get_num_parameters()
        if group == "backbone" or group == "fpn":
            adapter = self.backbone.add_adapter(group, reduction)
        elif group == "roi":
            adapter = self.detector.roi_heads.box_head.add_adapter(reduction)
        else:  # estimate_expansion_params already validates, retained for type narrowing
            raise ValueError(group)

        target_module = self._group_modules(group)[0]
        device, dtype = _module_device_and_dtype(target_module)
        adapter.to(device=device, dtype=dtype)
        added = self.get_num_parameters() - before
        if added != expected:
            raise RuntimeError(
                f"adapter estimate {expected} did not match actual delta {added}"
            )

        if group in self._freeze_snapshots and not self._freeze_preserve_adapters.get(
            group, False
        ):
            for parameter in adapter.parameters():
                self._freeze_snapshots[group].append(
                    (parameter, parameter.requires_grad)
                )
                parameter.requires_grad = False
            adapter.eval()
        return added

    def _group_modules(self, group: str) -> tuple[nn.Module, ...]:
        if group == "backbone":
            return self.backbone.body, self.backbone.backbone_adapters
        if group == "fpn":
            return self.backbone.fpn, self.backbone.fpn_adapters
        if group == "roi":
            return (
                self.detector.roi_heads.box_head,
                self.detector.roi_heads.box_predictor,
            )
        raise ValueError(f"group must be one of {EXPANDABLE_GROUPS}, got {group!r}")

    def _freeze_modules(self, group: str) -> tuple[nn.Module, ...]:
        if group == "backbone" and self._freeze_preserve_adapters.get(group, False):
            return (self.backbone.body,)
        return self._group_modules(group)

    def _group_parameters(self, group: str) -> Iterable[nn.Parameter]:
        seen: set[int] = set()
        for module in self._freeze_modules(group):
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter

    def freeze_group(
        self,
        group: str,
        *,
        until_step: int | None = None,
        preserve_adapters: bool = False,
    ) -> bool:
        """Freeze a group, optionally leaving backbone adapters trainable."""
        if until_step is not None and until_step < 0:
            raise ValueError("until_step must be non-negative")
        if preserve_adapters and group != "backbone":
            raise ValueError("preserve_adapters is supported only for backbone freezing")
        if group in self._freeze_snapshots:
            if self._freeze_preserve_adapters.get(group, False) != preserve_adapters:
                raise ValueError("Unfreeze the group before changing its freeze scope")
            self._frozen_until[group] = until_step
            for module in self._freeze_modules(group):
                module.eval()
            return False

        self._group_modules(group)  # Validate before changing freeze state.
        self._freeze_preserve_adapters[group] = preserve_adapters
        modules = list(self._freeze_modules(group))
        snapshot = [
            (parameter, parameter.requires_grad)
            for parameter in self._group_parameters(group)
        ]
        changed = any(previous for _, previous in snapshot)
        for parameter, _ in snapshot:
            parameter.requires_grad = False
        self._frozen_module_modes[group] = [
            (module, module.training) for module in modules
        ]
        for module in modules:
            module.eval()
        self._freeze_snapshots[group] = snapshot
        self._frozen_until[group] = until_step
        return changed

    def unfreeze_group(self, group: str) -> bool:
        """Restore the exact trainability state that preceded ``freeze_group``."""
        snapshot = self._freeze_snapshots.pop(group, None)
        if snapshot is None:
            return False
        changed = any(
            parameter.requires_grad != previous for parameter, previous in snapshot
        )
        for parameter, previous in snapshot:
            parameter.requires_grad = previous
        self._frozen_until.pop(group, None)
        self._freeze_preserve_adapters.pop(group, None)
        module_modes = self._frozen_module_modes.pop(group, ())
        if self.training:
            for module, previous in module_modes:
                module.train(previous)
        return changed

    def train(self, mode: bool = True):
        """Keep frozen groups in eval mode when the rest of the detector trains.

        ``requires_grad=False`` alone does not freeze BatchNorm running
        statistics. Reapplying eval mode here makes ``freeze_group`` a real
        functional freeze even after a normal ``model.train()`` call.
        """
        super().train(mode)
        if mode:
            for group in self._freeze_snapshots:
                for module in self._freeze_modules(group):
                    module.eval()
        return self

    def get_num_parameters(self, *, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            return sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            )
        return sum(parameter.numel() for parameter in parameters)

    def adapter_parameter_count(self, group: str | None = None) -> int:
        groups = EXPANDABLE_GROUPS if group is None else (group,)
        total = 0
        for name in groups:
            if name == "backbone":
                modules = self.backbone.backbone_adapters
            elif name == "fpn":
                modules = self.backbone.fpn_adapters
            elif name == "roi":
                modules = self.detector.roi_heads.box_head.adapters
            else:
                raise ValueError(
                    f"group must be one of {EXPANDABLE_GROUPS}, got {name!r}"
                )
            total += sum(parameter.numel() for parameter in modules.parameters())
        return total

    def architecture_manifest(self) -> dict[str, Any]:
        """Return the structure required before loading a dynamic state dict."""
        expansions: dict[str, list[dict[str, int]]] = {
            "backbone": [
                {"reduction": int(bank.reduction)}
                for bank in self.backbone.backbone_adapters
            ],
            "fpn": [
                {"reduction": int(bank.reduction)}
                for bank in self.backbone.fpn_adapters
            ],
            "roi": [
                {"reduction": int(adapter.reduction)}
                for adapter in self.detector.roi_heads.box_head.adapters
            ],
        }
        return {
            "format_version": 3,
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "adapter_reduction": self.adapter_reduction,
            "adapter_merge": self.adapter_merge,
            "class_balanced_roi_loss": self.class_balanced_roi_loss,
            "expansions": expansions,
            "frozen_groups": {
                group: self._frozen_until[group] for group in self._freeze_snapshots
            },
            "freeze_preserve_adapters": dict(self._freeze_preserve_adapters),
        }

    def apply_architecture_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Reconstruct adapter modules and freeze state before loading weights."""
        format_version = int(manifest.get("format_version", -1))
        if format_version not in {1, 2, 3}:
            raise ValueError("unsupported CARL-D architecture manifest version")
        if manifest.get("architecture") != self.architecture:
            raise ValueError(
                "manifest architecture does not match the constructed detector: "
                f"{manifest.get('architecture')!r} != {self.architecture!r}"
            )
        if int(manifest.get("num_classes", -1)) != self.num_classes:
            raise ValueError("manifest class count does not match this detector")
        requested_merge = str(
            manifest.get("adapter_merge", "sequential" if format_version == 1 else "")
        ).lower()
        if requested_merge != "parallel":
            raise ValueError("Sequential/unspecified adapter checkpoints require their historical code")
        self.adapter_merge = requested_merge
        self.backbone.set_adapter_merge(requested_merge)
        self.detector.roi_heads.box_head.set_adapter_merge(requested_merge)

        expansions = manifest.get("expansions")
        if not isinstance(expansions, Mapping):
            raise ValueError("manifest expansions must be a mapping")
        current = self.architecture_manifest()["expansions"]
        for group in EXPANDABLE_GROUPS:
            requested_entries = list(expansions.get(group, ()))
            existing_entries = current[group]
            if len(existing_entries) > len(requested_entries):
                raise ValueError(
                    f"cannot remove existing {group} adapters from a live model"
                )
            if existing_entries != requested_entries[: len(existing_entries)]:
                raise ValueError(
                    f"existing {group} adapters do not match manifest prefix"
                )
            for entry in requested_entries[len(existing_entries) :]:
                if not isinstance(entry, Mapping) or "reduction" not in entry:
                    raise ValueError(f"invalid {group} expansion entry")
                self.expand_group(group, int(entry["reduction"]))

        frozen_groups = manifest.get("frozen_groups", {})
        if not isinstance(frozen_groups, Mapping):
            raise ValueError("manifest frozen_groups must be a mapping")
        preserve = manifest.get("freeze_preserve_adapters", {})
        if not isinstance(preserve, Mapping) or any(
            group not in frozen_groups
            or not isinstance(value, bool)
            or (value and group != "backbone")
            for group, value in preserve.items()
        ):
            raise ValueError("invalid manifest freeze adapter policy")
        for group in tuple(self._freeze_snapshots):
            if group not in frozen_groups or self._freeze_preserve_adapters.get(
                group, False
            ) != preserve.get(group, False):
                self.unfreeze_group(group)
        for group, until_step in frozen_groups.items():
            if group not in EXPANDABLE_GROUPS:
                raise ValueError(f"invalid frozen group in manifest: {group!r}")
            until = None if until_step is None else int(until_step)
            self.freeze_group(
                group, until_step=until, preserve_adapters=preserve.get(group, False)
            )


def create_carl_d_detector(**kwargs: Any) -> CARLDDetector:
    """Construct the detection-only CARL-D model."""
    return CARLDDetector(**kwargs)
