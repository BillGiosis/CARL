"""Class-balanced loss with standard sampling for the shared Faster R-CNN head."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads


def _weighted_fastrcnn_loss(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    labels: list[torch.Tensor],
    regression_targets: list[torch.Tensor],
    class_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast R-CNN loss with bounded foreground-only class reweighting."""
    concatenated_labels = torch.cat(labels, dim=0)
    concatenated_targets = torch.cat(regression_targets, dim=0)
    sample_weights = class_weights.to(
        device=class_logits.device, dtype=class_logits.dtype
    )[concatenated_labels]
    per_proposal = F.cross_entropy(class_logits, concatenated_labels, reduction="none")
    classification_loss = (per_proposal * sample_weights).sum() / (
        sample_weights.sum().clamp_min(1.0)
    )

    positive = torch.where(concatenated_labels > 0)[0]
    positive_labels = concatenated_labels[positive]
    sample_count, _ = class_logits.shape
    box_regression = box_regression.reshape(
        sample_count, box_regression.size(-1) // 4, 4
    )
    box_loss = F.smooth_l1_loss(
        box_regression[positive, positive_labels],
        concatenated_targets[positive],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / max(1, concatenated_labels.numel())
    return classification_loss, box_loss


class ClassBalancedRoIHeads(RoIHeads):
    """Box-only RoI heads with configurable positive-proposal weights.

    Background proposals always retain unit weight. The supplied foreground
    weights are normalized and capped by the trainer before being installed.
    """

    @classmethod
    def from_roi_heads(cls, source: RoIHeads, num_classes: int):
        if source.has_mask() or source.has_keypoint():
            raise TypeError("CARL-D supports box-only Faster R-CNN RoI heads")
        result = cls(
            source.box_roi_pool,
            source.box_head,
            source.box_predictor,
            source.proposal_matcher.high_threshold,
            source.proposal_matcher.low_threshold,
            source.fg_bg_sampler.batch_size_per_image,
            source.fg_bg_sampler.positive_fraction,
            source.box_coder.weights,
            source.score_thresh,
            source.nms_thresh,
            source.detections_per_img,
        )
        result.register_buffer(
            "positive_class_weights",
            torch.ones(int(num_classes), dtype=torch.float32),
            persistent=False,
        )
        return result

    def set_positive_class_weights(self, weights: torch.Tensor) -> None:
        weights = torch.as_tensor(
            weights,
            device=self.positive_class_weights.device,
            dtype=torch.float32,
        )
        if weights.shape != self.positive_class_weights.shape:
            raise ValueError(
                "RoI class weights must contain background plus all foreground classes"
            )
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("RoI class weights must be finite and positive")
        if float(weights[0]) != 1.0:
            raise ValueError("background RoI weight must remain exactly one")
        self.positive_class_weights.copy_(weights)

    def forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None:
            for target in targets:
                if target["boxes"].dtype not in (torch.float, torch.double, torch.half):
                    raise TypeError("target boxes must have a floating-point dtype")
                if target["labels"].dtype != torch.int64:
                    raise TypeError("target labels must have dtype int64")

        if self.training:
            proposals, _, labels, regression_targets = self.select_training_samples(
                proposals, targets
            )
        else:
            labels = None
            regression_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        if self.training:
            if labels is None or regression_targets is None:
                raise RuntimeError("training RoI targets were not constructed")
            loss_classifier, loss_box_reg = _weighted_fastrcnn_loss(
                class_logits,
                box_regression,
                labels,
                regression_targets,
                self.positive_class_weights,
            )
            return [], {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
            }

        boxes, scores, predicted_labels = self.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes
        )
        return [
            {"boxes": box, "labels": label, "scores": score}
            for box, score, label in zip(boxes, scores, predicted_labels)
        ], {}
