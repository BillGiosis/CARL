"""Detection metrics for the four-domain CLAD-D continual-learning stream.

The detector has seven category logits (background plus six foreground
categories), but detection metrics are defined only for foreground labels
1..6.  This module rejects label 0 instead of silently including background in
the mAP denominator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union

import torch

DEFAULT_FOREGROUND_CLASS_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)


def _mean_finite(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else math.nan


def _scalar_metric(value: torch.Tensor) -> float:
    number = float(value.detach().cpu().item())
    return number if number >= 0.0 else math.nan


def _validate_foreground_ids(class_ids: Sequence[int]) -> Tuple[int, ...]:
    normalized = tuple(int(class_id) for class_id in class_ids)
    if not normalized:
        raise ValueError("foreground_class_ids cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("foreground_class_ids must be unique")
    if 0 in normalized:
        raise ValueError("background label 0 cannot be a foreground class")
    return normalized


def _prepare_sample(
    sample: Mapping[str, torch.Tensor],
    *,
    prediction: bool,
    foreground_class_ids: Tuple[int, ...],
) -> Dict[str, torch.Tensor]:
    kind = "prediction" if prediction else "target"
    required = {"boxes", "labels", "scores"} if prediction else {"boxes", "labels"}
    missing = required.difference(sample)
    if missing:
        raise ValueError(f"{kind} is missing fields: {sorted(missing)}")

    boxes = (
        torch.as_tensor(sample["boxes"])
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .clone()
    )
    labels = (
        torch.as_tensor(sample["labels"])
        .detach()
        .to(device="cpu", dtype=torch.int64)
        .clone()
    )

    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"{kind} boxes must have shape [N, 4]")
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise ValueError(f"{kind} labels must have shape [N] and align with boxes")
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{kind} boxes must be finite")
    if (
        prediction
        and boxes.numel()
        and ((boxes[:, 2] <= boxes[:, 0]).any() or (boxes[:, 3] <= boxes[:, 1]).any())
    ):
        raise ValueError("prediction boxes must have positive width and height")
    allowed = torch.tensor(foreground_class_ids, dtype=torch.int64)
    if labels.numel() and not torch.isin(labels, allowed).all():
        invalid = sorted(set(labels.tolist()).difference(foreground_class_ids))
        raise ValueError(
            f"{kind} contains non-foreground labels {invalid}; "
            "CLAD-D metrics accept labels 1..6 and exclude background label 0"
        )

    prepared: Dict[str, torch.Tensor] = {"boxes": boxes, "labels": labels}
    if prediction:
        scores = (
            torch.as_tensor(sample["scores"])
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .clone()
        )
        if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
            raise ValueError(
                "prediction scores must have shape [N] and align with boxes"
            )
        if not torch.isfinite(scores).all():
            raise ValueError("prediction scores must be finite")
        prepared["scores"] = scores
    else:
        for optional_field, dtype in (
            ("area", torch.float32),
            ("iscrowd", torch.int64),
        ):
            if optional_field in sample:
                value = (
                    torch.as_tensor(sample[optional_field])
                    .detach()
                    .to(device="cpu", dtype=dtype)
                    .clone()
                )
                if value.ndim != 1 or value.shape[0] != boxes.shape[0]:
                    raise ValueError(
                        f"target {optional_field} must have shape [N] and align "
                        "with boxes"
                    )
                prepared[optional_field] = value
    return prepared


@dataclass(frozen=True)
class AP50Result:
    """Per-domain COCO 101-point bounding-box AP at IoU 0.50."""

    map50: float
    per_class_ap50: Dict[int, float]
    target_count_per_class: Dict[int, int]
    num_images: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "map50": self.map50,
            "per_class_ap50": dict(self.per_class_ap50),
            "target_count_per_class": dict(self.target_count_per_class),
            "num_images": self.num_images,
        }


def evaluate_ap50(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    targets: Sequence[Mapping[str, torch.Tensor]],
    foreground_class_ids: Sequence[int] = DEFAULT_FOREGROUND_CLASS_IDS,
) -> AP50Result:
    """Evaluate COCO-style AP50 without treating background as a class.

    ``predictions`` and ``targets`` use the Torchvision detector convention.
    Scores are returned in the 0..1 range. A class with no ground-truth object
    receives ``NaN`` in ``per_class_ap50`` and is not added to the mAP
    denominator, matching COCO evaluation behavior.
    """

    class_ids = _validate_foreground_ids(foreground_class_ids)
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must contain the same images")
    if not targets:
        raise ValueError("at least one image is required for AP evaluation")

    prepared_predictions = [
        _prepare_sample(sample, prediction=True, foreground_class_ids=class_ids)
        for sample in predictions
    ]
    prepared_targets = [
        _prepare_sample(sample, prediction=False, foreground_class_ids=class_ids)
        for sample in targets
    ]

    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError as exc:  # pragma: no cover - exercised by deployment only
        raise RuntimeError(
            "Detection AP requires torchmetrics and pycocotools"
        ) from exc

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[0.5],
        class_metrics=True,
    )
    metric.update(prepared_predictions, prepared_targets)
    raw = metric.compute()

    per_class = {class_id: math.nan for class_id in class_ids}
    raw_classes = torch.atleast_1d(raw["classes"]).detach().cpu().tolist()
    raw_values = torch.atleast_1d(raw["map_per_class"]).detach().cpu().tolist()
    for class_id, value in zip(raw_classes, raw_values):
        class_id = int(class_id)
        if class_id in per_class and float(value) >= 0.0:
            per_class[class_id] = float(value)

    target_counts = {class_id: 0 for class_id in class_ids}
    for target in prepared_targets:
        for class_id in class_ids:
            target_counts[class_id] += int((target["labels"] == class_id).sum())

    return AP50Result(
        map50=_scalar_metric(raw["map_50"]),
        per_class_ap50=per_class,
        target_count_per_class=target_counts,
        num_images=len(targets),
    )


@dataclass(frozen=True)
class DomainAP50Summary:
    """AP50 results kept separate per domain before equal-domain averaging."""

    per_domain: Dict[Hashable, AP50Result]
    equal_domain_map50: float
    per_class_equal_domain_ap50: Dict[int, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "per_domain": {
                domain: result.as_dict() for domain, result in self.per_domain.items()
            },
            "equal_domain_map50": self.equal_domain_map50,
            "per_class_equal_domain_ap50": dict(self.per_class_equal_domain_ap50),
        }




def evaluate_environment_ap50(
    predictions_by_domain: Mapping[Hashable, Sequence[Mapping[str, torch.Tensor]]],
    targets_by_domain: Mapping[Hashable, Sequence[Mapping[str, torch.Tensor]]],
    image_metadata: Mapping[int, Mapping[str, Any]],
    foreground_class_ids: Sequence[int] = DEFAULT_FOREGROUND_CLASS_IDS,
) -> Dict[str, Any]:
    """Recompute subgroup AP from shared predictions, never average domain APs."""
    groups = {
        "Day": ("period", "Daytime"),
        "Night": ("period", "Night"),
        "Highway": ("location", "Highway"),
        "City": ("location", "Citystreet"),
        "Country": ("location", "Countryroad"),
    }
    predictions = {name: [] for name in groups}
    targets = {name: [] for name in groups}
    seen = set()
    for domain, domain_targets in targets_by_domain.items():
        for prediction, target in zip(
            predictions_by_domain[domain], domain_targets, strict=True
        ):
            image_id = int(torch.as_tensor(target["image_id"]).item())
            if image_id in seen:
                raise ValueError(f"Duplicate test image in environmental evaluation: {image_id}")
            seen.add(image_id)
            metadata = image_metadata[image_id]
            for name, (field, value) in groups.items():
                if metadata[field] == value:
                    predictions[name].append(prediction)
                    targets[name].append(target)
    return {
        name: (
            evaluate_ap50(predictions[name], targets[name], foreground_class_ids).as_dict()
            if targets[name] else AP50Result(
                math.nan, {c: math.nan for c in foreground_class_ids},
                {c: 0 for c in foreground_class_ids}, 0,
            ).as_dict()
        )
        for name in groups
    }


def evaluate_domains_ap50(
    predictions_by_domain: Mapping[Hashable, Sequence[Mapping[str, torch.Tensor]]],
    targets_by_domain: Mapping[Hashable, Sequence[Mapping[str, torch.Tensor]]],
    foreground_class_ids: Sequence[int] = DEFAULT_FOREGROUND_CLASS_IDS,
) -> DomainAP50Summary:
    """Evaluate each domain independently and then macro-average domains."""

    class_ids = _validate_foreground_ids(foreground_class_ids)
    if set(predictions_by_domain) != set(targets_by_domain):
        raise ValueError("prediction and target domain keys must match")
    if not targets_by_domain:
        raise ValueError("at least one domain is required")

    per_domain = {
        domain: evaluate_ap50(
            predictions_by_domain[domain],
            targets_by_domain[domain],
            foreground_class_ids=class_ids,
        )
        for domain in targets_by_domain
    }
    per_class_macro = {
        class_id: _mean_finite(
            [result.per_class_ap50[class_id] for result in per_domain.values()]
        )
        for class_id in class_ids
    }
    return DomainAP50Summary(
        per_domain=per_domain,
        equal_domain_map50=_mean_finite(
            [result.map50 for result in per_domain.values()]
        ),
        per_class_equal_domain_ap50=per_class_macro,
    )




class DomainAP50History:
    """Track the task-by-evaluation-domain AP50 matrix.

    Rows represent the task most recently trained and columns represent the
    domain evaluated. Unobserved entries remain ``NaN`` so future-domain scores
    cannot be confused with zero performance.
    """

    def __init__(self, num_domains: int = 4):
        if num_domains <= 0:
            raise ValueError("num_domains must be positive")
        self.num_domains = int(num_domains)
        self.matrix = torch.full(
            (self.num_domains, self.num_domains),
            math.nan,
            dtype=torch.float64,
        )

    def record(
        self,
        after_task: int,
        per_domain: Mapping[int, Union[AP50Result, float]],
    ) -> None:
        after_task = int(after_task)
        if not 0 <= after_task < self.num_domains:
            raise ValueError("after_task is outside the configured stream")
        if not per_domain:
            raise ValueError("per_domain cannot be empty")
        for domain, result in per_domain.items():
            domain = int(domain)
            if not 0 <= domain < self.num_domains:
                raise ValueError(f"domain {domain} is outside the configured stream")
            if domain > after_task:
                raise ValueError("future-domain metrics cannot be recorded")
            value = result.map50 if isinstance(result, AP50Result) else float(result)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("AP50 values must be finite and in the 0..1 range")
            self.matrix[after_task, domain] = value

    @property
    def latest_task(self) -> Optional[int]:
        populated_rows = torch.isfinite(self.matrix).any(dim=1).nonzero()
        return int(populated_rows[-1]) if populated_rows.numel() else None

    def equal_domain_map50(self, after_task: Optional[int] = None) -> float:
        if after_task is None:
            after_task = self.latest_task
        if after_task is None:
            return math.nan
        values = self.matrix[int(after_task), : int(after_task) + 1]
        finite = values[torch.isfinite(values)]
        return float(finite.mean()) if finite.numel() else math.nan

    def anytime_equal_domain_map50(self) -> float:
        row_means = [
            self.equal_domain_map50(task)
            for task in range(self.num_domains)
            if torch.isfinite(self.matrix[task]).any()
        ]
        return _mean_finite(row_means)

    def forgetting(self) -> float:
        final_task = self.latest_task
        if final_task is None or final_task == 0:
            return 0.0
        values = []
        for domain in range(final_task):
            history = self.matrix[domain : final_task + 1, domain]
            history = history[torch.isfinite(history)]
            if history.numel() >= 2:
                values.append(float(history.max() - history[-1]))
        return _mean_finite(values) if values else 0.0

    def backward_transfer(self) -> float:
        final_task = self.latest_task
        if final_task is None or final_task == 0:
            return 0.0
        values = []
        for domain in range(final_task):
            learned = self.matrix[domain, domain]
            final = self.matrix[final_task, domain]
            if torch.isfinite(learned) and torch.isfinite(final):
                values.append(float(final - learned))
        return _mean_finite(values) if values else 0.0

    def plasticity(self) -> float:
        diagonal = torch.diagonal(self.matrix)
        finite = diagonal[torch.isfinite(diagonal)]
        return float(finite.mean()) if finite.numel() else math.nan

    def summary(self) -> Dict[str, Any]:
        return {
            "ap50_matrix": self.matrix.clone(),
            "latest_task": self.latest_task,
            "equal_domain_map50": self.equal_domain_map50(),
            "anytime_equal_domain_map50": self.anytime_equal_domain_map50(),
            "forgetting": self.forgetting(),
            "backward_transfer": self.backward_transfer(),
            "plasticity": self.plasticity(),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "num_domains": self.num_domains,
            "matrix": self.matrix.clone(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if int(state_dict["num_domains"]) != self.num_domains:
            raise ValueError("history num_domains does not match")
        matrix = torch.as_tensor(state_dict["matrix"], dtype=torch.float64)
        if matrix.shape != self.matrix.shape:
            raise ValueError("history matrix has the wrong shape")
        self.matrix.copy_(matrix)
