"""Image-level replay storage for CLAD-D object detection.

An entry represents one complete image and all of its annotations. Capacity is
therefore measured in unique images, not objects or cropped object instances.
"""

from __future__ import annotations

import copy
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
from torch.utils._pytree import tree_map

CLADD_MAX_REPLAY_IMAGES = 250
DEFAULT_FOREGROUND_CLASS_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
_SELECTION_SCORE_VALID_KEY = "_replay_selection_score_valid"
_REFINEMENT_SCORE_METADATA_KEYS: Tuple[str, ...] = (
    "refinement_foreground_entropy",
    "refinement_localization_error",
    "refinement_rare_class_gain",
)


def _clone_cpu(value: Any) -> Any:
    return tree_map(
        lambda item: (
            item.detach().to(device="cpu").clone()
            if torch.is_tensor(item)
            else copy.deepcopy(item)
        ),
        value,
    )


def _to_device(value: Any, device: torch.device) -> Any:
    return tree_map(
        lambda item: (
            item.to(device=device, non_blocking=True) if torch.is_tensor(item) else item
        ),
        value,
    )


def _normalize_image_id(image_id: Any) -> Hashable:
    if torch.is_tensor(image_id):
        if image_id.numel() != 1:
            raise ValueError("image_id tensors must contain exactly one value")
        image_id = image_id.detach().cpu().item()
    try:
        hash(image_id)
    except TypeError as exc:
        raise ValueError("image_id must be hashable") from exc
    return image_id


def _validate_target(
    target: Mapping[str, Any], foreground_class_ids: Tuple[int, ...]
) -> Dict[str, Any]:
    missing = {"boxes", "labels"}.difference(target)
    if missing:
        raise ValueError(f"detection target is missing fields: {sorted(missing)}")
    cloned = _clone_cpu(dict(target))
    boxes = torch.as_tensor(cloned["boxes"], dtype=torch.float32).clone()
    labels = torch.as_tensor(cloned["labels"], dtype=torch.int64).clone()
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("target boxes must have shape [N, 4]")
    if labels.ndim != 1 or len(labels) != len(boxes):
        raise ValueError("target labels must have shape [N] and align with boxes")
    if not torch.isfinite(boxes).all():
        raise ValueError("target boxes must be finite")
    allowed = torch.tensor(foreground_class_ids, dtype=torch.int64)
    if labels.numel() and not torch.isin(labels, allowed).all():
        invalid = sorted(set(labels.tolist()).difference(foreground_class_ids))
        raise ValueError(
            f"target contains invalid foreground labels {invalid}; expected "
            f"{list(foreground_class_ids)}"
        )
    cloned["boxes"] = boxes
    cloned["labels"] = labels
    return cloned


def _size_bins(target: Mapping[str, Any]) -> Tuple[str, ...]:
    if "area" in target:
        area = torch.as_tensor(target["area"], dtype=torch.float32).reshape(-1)
    else:
        boxes = torch.as_tensor(target["boxes"], dtype=torch.float32)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    bins = set()
    for value in area.tolist():
        if value < 32**2:
            bins.add("small")
        elif value < 96**2:
            bins.add("medium")
        else:
            bins.add("large")
    return tuple(name for name in ("small", "medium", "large") if name in bins)


@dataclass
class DetectionReplayRecord:
    """One unique full image and its complete detection annotation."""

    image_id: Hashable
    domain_id: int
    target: Dict[str, Any]
    image: Optional[torch.Tensor] = None
    image_path: Optional[str] = None
    classes: Tuple[int, ...] = ()
    size_bins: Tuple[str, ...] = ()
    selection_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    @property
    def key(self) -> Tuple[int, Hashable]:
        return self.domain_id, self.image_id

    def clone(self) -> "DetectionReplayRecord":
        return DetectionReplayRecord(
            image_id=copy.deepcopy(self.image_id),
            domain_id=self.domain_id,
            target=_clone_cpu(self.target),
            image=_clone_cpu(self.image),
            image_path=self.image_path,
            classes=tuple(self.classes),
            size_bins=tuple(self.size_bins),
            selection_score=float(self.selection_score),
            metadata=_clone_cpu(self.metadata or {}),
        )


class DetectionReplayBuffer:
    """Adaptive, image-level replay buffer with a hard 250-image ceiling.

    The standard ``add`` path retains seeded reservoir semantics. Completed
    domains can instead be admitted jointly with soft multi-label, domain, and
    object-scale balancing via ``admit_balanced_candidates``. Refinement keeps
    detector-derived utility and feature diversity separate from admission,
    while replay sampling can balance both the stored population and each
    sampled mini-batch.
    """

    STATE_VERSION = 2

    def __init__(
        self,
        initial_capacity: int = 50,
        max_capacity: int = CLADD_MAX_REPLAY_IMAGES,
        seed: int = 0,
        foreground_class_ids: Sequence[int] = DEFAULT_FOREGROUND_CLASS_IDS,
    ):
        max_capacity = int(max_capacity)
        initial_capacity = int(initial_capacity)
        if not 1 <= max_capacity <= CLADD_MAX_REPLAY_IMAGES:
            raise ValueError(f"max_capacity must be in 1..{CLADD_MAX_REPLAY_IMAGES}")
        if not 0 <= initial_capacity <= max_capacity:
            raise ValueError("initial_capacity must be in 0..max_capacity")
        class_ids = tuple(int(class_id) for class_id in foreground_class_ids)
        if not class_ids or 0 in class_ids or len(set(class_ids)) != len(class_ids):
            raise ValueError(
                "foreground_class_ids must be unique, nonempty, and exclude 0"
            )

        self.max_capacity = max_capacity
        self.current_capacity = initial_capacity
        self.foreground_class_ids = class_ids
        self._records: List[DetectionReplayRecord] = []
        self._record_indices: Dict[Tuple[int, Hashable], int] = {}
        self._seen_keys: set[Tuple[int, Hashable]] = set()
        self._unique_insertions = 0
        self._sampling_cache: Dict[tuple, tuple] = {}
        self._generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def _canonical_record(
        self,
        *,
        image_id: Any,
        domain_id: int,
        target: Mapping[str, Any],
        image: Optional[torch.Tensor] = None,
        image_path: Optional[os.PathLike] = None,
        selection_score: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> DetectionReplayRecord:
        image_id = _normalize_image_id(image_id)
        domain_id = int(domain_id)
        if domain_id < 0:
            raise ValueError("domain_id must be nonnegative")
        if image is None and image_path is None:
            raise ValueError("a replay record needs either image or image_path")
        if image is not None:
            if not torch.is_tensor(image):
                raise TypeError("image must be a tensor when provided")
            if image.ndim != 3:
                raise ValueError("full detection images must have shape [C, H, W]")
            image = image.detach().to(device="cpu").clone()
        path = os.fspath(image_path) if image_path is not None else None
        target_copy = _validate_target(target, self.foreground_class_ids)
        classes = tuple(sorted(set(target_copy["labels"].tolist())))
        selection_score = float(selection_score)
        if not math.isfinite(selection_score):
            raise ValueError("selection_score must be finite")
        return DetectionReplayRecord(
            image_id=image_id,
            domain_id=domain_id,
            target=target_copy,
            image=image,
            image_path=path,
            classes=classes,
            size_bins=_size_bins(target_copy),
            selection_score=selection_score,
            metadata=_clone_cpu(dict(metadata or {})),
        )

    def add(
        self,
        *,
        image_id: Any,
        domain_id: int,
        target: Mapping[str, Any],
        image: Optional[torch.Tensor] = None,
        image_path: Optional[os.PathLike] = None,
        selection_score: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Add/update a sample; return whether the sample is retained."""
        self._sampling_cache.clear()
        normalized_key = (int(domain_id), _normalize_image_id(image_id))
        existing = self._record_indices.get(normalized_key)
        if existing is None and normalized_key in self._seen_keys:
            # Repeated epochs are not repeated stream samples. A physical
            # image receives exactly one reservoir draw even if it was not
            # retained by that draw.
            return False
        record = self._canonical_record(
            image_id=image_id,
            domain_id=domain_id,
            target=target,
            image=image,
            image_path=image_path,
            selection_score=selection_score,
            metadata=metadata,
        )
        if existing is not None:
            self._records[existing] = record
            return True

        self._seen_keys.add(record.key)
        self._unique_insertions += 1
        if self.current_capacity == 0:
            return False
        if len(self._records) < self.current_capacity:
            self._records.append(record)
            self._record_indices[record.key] = len(self._records) - 1
            return True

        replacement = int(
            torch.randint(
                self._unique_insertions,
                size=(1,),
                generator=self._generator,
            ).item()
        )
        if replacement >= self.current_capacity:
            return False
        old_key = self._records[replacement].key
        del self._record_indices[old_key]
        self._records[replacement] = record
        self._record_indices[record.key] = replacement
        return True

    def add_many(self, samples: Iterable[Mapping[str, Any]]) -> int:
        retained = 0
        for sample in samples:
            retained += int(self.add(**dict(sample)))
        return retained

    @staticmethod
    def _effective_number_weights(
        counts: Mapping[Any, int],
        *,
        beta: float,
        max_weight: float,
    ) -> Dict[Any, float]:
        keys = tuple(counts)
        if not keys:
            return {}
        values = torch.tensor(
            [max(1, int(counts[key])) for key in keys], dtype=torch.float64
        )
        raw = (1.0 - float(beta)) / (
            1.0 - torch.pow(torch.full_like(values, float(beta)), values)
        )
        raw.div_(raw.mean().clamp_min(torch.finfo(raw.dtype).eps))
        raw.clamp_(min=1.0 / float(max_weight), max=float(max_weight))
        return {key: float(weight) for key, weight in zip(keys, raw.tolist())}

    def _hierarchical_admission_indices(
        self,
        records: Sequence[DetectionReplayRecord],
        count: int,
        *,
        balance_strength: float,
        class_balance_beta: float,
        max_sample_weight: float,
        utility_sampling_strength: float,
    ) -> List[int]:
        """Select domains first, then balance classes within each domain.

        Domain probability is independent of the number of candidate images in
        that domain, preventing a newly completed domain from dominating the
        buffer merely because all of its images are available. Selection stays
        stochastic and quota-free. After choosing a domain, image weights use
        multi-label class and scale deficits measured within that domain, plus
        the bounded detector-utility multiplier used by replay sampling.
        """

        count = min(max(0, int(count)), len(records))
        if count == 0:
            return []

        available_by_domain: Dict[int, List[int]] = {}
        for index, record in enumerate(records):
            available_by_domain.setdefault(record.domain_id, []).append(index)

        class_weights_by_domain: Dict[int, Dict[Any, float]] = {}
        size_weights_by_domain: Dict[int, Dict[Any, float]] = {}
        for domain_id, indices in available_by_domain.items():
            class_counts = Counter(
                class_id for index in indices for class_id in records[index].classes
            )
            size_counts = Counter(
                size_name for index in indices for size_name in records[index].size_bins
            )
            class_weights_by_domain[domain_id] = self._effective_number_weights(
                class_counts,
                beta=class_balance_beta,
                max_weight=max_sample_weight,
            )
            size_weights_by_domain[domain_id] = self._effective_number_weights(
                size_counts,
                beta=class_balance_beta,
                max_weight=max_sample_weight,
            )

        utility_weights = self._utility_sampling_weights(
            records, float(utility_sampling_strength)
        )

        selected: List[int] = []
        selected_domains: Counter = Counter()
        selected_classes_by_domain: Dict[int, Counter] = {
            domain_id: Counter() for domain_id in available_by_domain
        }
        selected_sizes_by_domain: Dict[int, Counter] = {
            domain_id: Counter() for domain_id in available_by_domain
        }
        strength = float(balance_strength)

        while available_by_domain and len(selected) < count:
            # The domain draw has one entry per domain rather than one entry
            # per image. Candidate-pool size therefore cannot dominate its
            # probability mass. The diminishing term softly approaches equal
            # domain representation without imposing exact quotas.
            domain_ids = list(available_by_domain)
            domain_weights = torch.tensor(
                [1.0 / (1.0 + selected_domains[domain_id]) for domain_id in domain_ids],
                dtype=torch.float64,
            )
            domain_weights.div_(
                domain_weights.mean().clamp_min(torch.finfo(domain_weights.dtype).eps)
            )
            domain_weights.mul_(strength).add_(1.0 - strength)
            domain_local_index = int(
                torch.multinomial(
                    domain_weights, 1, replacement=False, generator=self._generator
                ).item()
            )
            domain_id = domain_ids[domain_local_index]
            domain_indices = available_by_domain[domain_id]
            selected_classes = selected_classes_by_domain[domain_id]
            selected_sizes = selected_sizes_by_domain[domain_id]
            class_weights = class_weights_by_domain[domain_id]
            size_weights = size_weights_by_domain[domain_id]

            marginal = []
            for index in domain_indices:
                record = records[index]
                class_gain = max(
                    (
                        class_weights.get(class_id, 1.0)
                        / math.sqrt(1.0 + selected_classes[class_id])
                        for class_id in record.classes
                    ),
                    default=0.0,
                )
                size_gain = max(
                    (
                        size_weights.get(size_name, 1.0)
                        / math.sqrt(1.0 + selected_sizes[size_name])
                        for size_name in record.size_bins
                    ),
                    default=0.0,
                )
                marginal.append(class_gain + 0.25 * size_gain)

            image_weights = torch.tensor(marginal, dtype=torch.float64)
            image_weights.div_(
                image_weights.mean().clamp_min(torch.finfo(image_weights.dtype).eps)
            )
            image_weights.clamp_(max=float(max_sample_weight))
            image_weights.mul_(strength).add_(1.0 - strength)
            image_weights.mul_(utility_weights[domain_indices])
            image_local_index = int(
                torch.multinomial(
                    image_weights, 1, replacement=False, generator=self._generator
                ).item()
            )
            record_index = domain_indices.pop(image_local_index)
            selected.append(record_index)
            record = records[record_index]
            selected_domains[domain_id] += 1
            selected_classes.update(record.classes)
            selected_sizes.update(record.size_bins)
            if not domain_indices:
                del available_by_domain[domain_id]
        return selected

    def admit_balanced_candidates(
        self,
        candidates: Iterable[Mapping[str, Any]],
        *,
        balance_strength: float,
        class_balance_beta: float,
        max_sample_weight: float,
        utility_sampling_strength: float = 0.0,
    ) -> bool:
        """Admit a completed domain with hierarchical soft balancing.

        Previously retained images are considered together with every new
        completed-domain candidate. Old images that were not retained are not
        revisited. Every new physical image is marked seen exactly once, and
        the active capacity remains unchanged.
        """

        if not 0.0 <= float(balance_strength) <= 1.0:
            raise ValueError("balance_strength must be in [0, 1]")
        if (
            not math.isfinite(float(class_balance_beta))
            or not 0.0 <= float(class_balance_beta) < 1.0
        ):
            raise ValueError("class_balance_beta must be in [0, 1)")
        if (
            not math.isfinite(float(max_sample_weight))
            or float(max_sample_weight) < 1.0
        ):
            raise ValueError("max_sample_weight must be finite and at least one")
        if (
            not math.isfinite(float(utility_sampling_strength))
            or not 0.0 <= float(utility_sampling_strength) <= 1.0
        ):
            raise ValueError("utility_sampling_strength must be in [0, 1]")

        canonical_candidates: Dict[Tuple[int, Hashable], DetectionReplayRecord] = {}
        for sample in candidates:
            if not isinstance(sample, Mapping):
                raise TypeError("each replay candidate must be a sample mapping")
            record = self._canonical_record(**dict(sample))
            canonical_candidates[record.key] = record

        old_keys = [record.key for record in self._records]
        merged_by_key: Dict[Tuple[int, Hashable], DetectionReplayRecord] = {
            record.key: record.clone() for record in self._records
        }
        merged_by_key.update(canonical_candidates)
        merged = list(merged_by_key.values())
        keep_count = min(self.current_capacity, len(merged))
        selected_indices = self._hierarchical_admission_indices(
            merged,
            keep_count,
            balance_strength=float(balance_strength),
            class_balance_beta=float(class_balance_beta),
            max_sample_weight=float(max_sample_weight),
            utility_sampling_strength=float(utility_sampling_strength),
        )
        retained = [merged[index] for index in selected_indices]

        new_seen_keys = set(canonical_candidates).difference(self._seen_keys)
        self._seen_keys.update(canonical_candidates)
        self._unique_insertions += len(new_seen_keys)
        self._records = retained
        self._rebuild_index()
        return old_keys != [record.key for record in retained]

    @staticmethod
    def _score_for(
        record: DetectionReplayRecord,
        scores: Optional[Mapping[Any, float]],
        score_fn: Optional[Callable[[DetectionReplayRecord], float]],
    ) -> float:
        if score_fn is not None:
            score = float(score_fn(record.clone()))
        elif scores is not None:
            if record.key in scores:
                score = float(scores[record.key])
            elif record.image_id in scores:
                score = float(scores[record.image_id])
            else:
                score = float(record.selection_score)
        else:
            score = float(record.selection_score)
        if not math.isfinite(score):
            raise ValueError(
                f"selection score for replay key {record.key!r} is not finite"
            )
        return score

    @staticmethod
    def _feature_vector(record: DetectionReplayRecord) -> Optional[torch.Tensor]:
        raw = (record.metadata or {}).get("refinement_feature")
        if raw is None:
            return None
        feature = torch.as_tensor(raw, dtype=torch.float32).reshape(-1)
        if not feature.numel() or not torch.isfinite(feature).all():
            return None
        return feature / feature.norm(p=2).clamp_min(1e-12)

    def _selection_utility(
        self,
        records: Sequence[DetectionReplayRecord],
        *,
        redundancy_weight: float,
    ) -> float:
        utility = sum(float(record.selection_score) for record in records)
        if redundancy_weight <= 0.0 or len(records) < 2:
            return utility
        features = [self._feature_vector(record) for record in records]
        if any(feature is None for feature in features):
            return utility
        if (
            len({int(feature.numel()) for feature in features if feature is not None})
            != 1
        ):
            return utility
        matrix = torch.stack([feature for feature in features if feature is not None])
        similarities = matrix @ matrix.T
        similarities.fill_diagonal_(-torch.inf)
        redundancy = similarities.max(dim=1).values.clamp_min(0.0).sum().item()
        return utility - redundancy_weight * float(redundancy)

    @staticmethod
    def _has_selection_score(record: DetectionReplayRecord) -> bool:
        """Return whether a record has a meaningful refinement utility.

        Records admitted through ordinary reservoir sampling retain the neutral
        default score of zero. Refinement metadata (or a nonzero score from an
        older checkpoint/caller) distinguishes evaluated records without
        requiring a replay-state format change.
        """

        metadata = record.metadata or {}
        return (
            bool(metadata.get(_SELECTION_SCORE_VALID_KEY, False))
            or any(key in metadata for key in _REFINEMENT_SCORE_METADATA_KEYS)
            or float(record.selection_score) != 0.0
        )

    @classmethod
    def _utility_sampling_weights(
        cls,
        records: Sequence[DetectionReplayRecord],
        strength: float,
    ) -> torch.Tensor:
        """Build bounded, rank-based utility multipliers for replay draws.

        A conceptual zero-score record anchors neutral utility. Evaluated
        records rank above or below that anchor, while records that have never
        been evaluated receive exactly unit weight. The bounded multipliers
        keep old refinement scores from dominating newer class/domain balance
        information between refinement decisions.
        """

        multipliers = torch.ones(len(records), dtype=torch.float64)
        scored = [
            (index, float(record.selection_score))
            for index, record in enumerate(records)
            if cls._has_selection_score(record)
        ]
        if not scored or strength <= 0.0:
            return multipliers

        reference_values = [0.0, *(score for _, score in scored)]
        reference_count = float(len(reference_values))

        def middle_quantile(value: float) -> float:
            below = sum(candidate < value for candidate in reference_values)
            equal = sum(candidate == value for candidate in reference_values)
            return (float(below) + 0.5 * float(equal)) / reference_count

        neutral_quantile = middle_quantile(0.0)
        for index, score in scored:
            relative_rank = 2.0 * (middle_quantile(score) - neutral_quantile)
            relative_rank = max(-1.0, min(1.0, relative_rank))
            # Even at strength=1, a stale score changes a record's weight by
            # at most 50%, rather than overwhelming the sampling distribution.
            multipliers[index] = 1.0 + 0.5 * strength * relative_rank
        return multipliers

    def _utility_refined_indices(
        self,
        records: Sequence[DetectionReplayRecord],
        count: int,
        *,
        redundancy_weight: float,
    ) -> List[int]:
        """Select by detection utility while discouraging feature redundancy.

        Buffer composition is deliberately not governed by hand-written image
        quotas. Class imbalance is handled when current and replay images are
        sampled and in the positive RoI classification loss.
        """
        selected: List[int] = []
        available = set(range(len(records)))
        feature_vectors = [self._feature_vector(record) for record in records]
        similarity_matrix: Optional[torch.Tensor] = None
        if feature_vectors and all(feature is not None for feature in feature_vectors):
            dimensions = {
                int(feature.numel())
                for feature in feature_vectors
                if feature is not None
            }
            if len(dimensions) == 1:
                matrix = torch.stack(
                    [feature for feature in feature_vectors if feature is not None]
                )
                similarity_matrix = (matrix @ matrix.T).clamp_min(0.0)
        maximum_redundancy = torch.zeros(len(records), dtype=torch.float32)

        def select(index: int) -> None:
            available.remove(index)
            selected.append(index)
            if similarity_matrix is not None:
                maximum_redundancy.copy_(
                    torch.maximum(maximum_redundancy, similarity_matrix[:, index])
                )

        def adjusted_utility(index: int) -> float:
            value = float(records[index].selection_score)
            if redundancy_weight <= 0.0 or not selected or similarity_matrix is None:
                return value
            return value - redundancy_weight * float(maximum_redundancy[index])

        while available and len(selected) < count:
            best = max(
                available,
                key=lambda index: (adjusted_utility(index), -index),
            )
            select(best)
        return selected

    def refine_with_candidates(
        self,
        candidates: Iterable[Mapping[str, Any]],
        *,
        new_capacity: Optional[int] = None,
        scores: Optional[Mapping[Any, float]] = None,
        score_fn: Optional[Callable[[DetectionReplayRecord], float]] = None,
        feature_redundancy_weight: float = 0.0,
    ) -> bool:
        """Refine or atomically resize over retained and candidate images.

        Candidate mappings use the same fields as :meth:`add`. All records are
        canonicalized before state is changed, duplicate ``(domain_id,
        image_id)`` candidates collapse to their last value, and a candidate
        replaces the retained value for the same key. Selection uses
        ``scores``, ``score_fn``, or each record's ``selection_score`` together
        with feature diversity. Capacity remains fixed unless ``new_capacity``
        is supplied. Class balance is applied by the replay sampler instead of
        fixed per-class or per-domain image quotas.

        Every unique candidate key is recorded as seen exactly once, including
        candidates that are not selected. Repeating a refinement pool therefore
        neither creates additional stream insertions nor duplicate records.
        """

        if scores is not None and score_fn is not None:
            raise ValueError("provide scores or score_fn, not both")
        target_capacity = (
            self.current_capacity if new_capacity is None else int(new_capacity)
        )
        if not 0 <= target_capacity <= self.max_capacity:
            raise ValueError("new_capacity must be in 0..max_capacity")
        if (
            not math.isfinite(float(feature_redundancy_weight))
            or float(feature_redundancy_weight) < 0.0
        ):
            raise ValueError("feature_redundancy_weight must be finite and nonnegative")

        canonical_candidates: Dict[Tuple[int, Hashable], DetectionReplayRecord] = {}
        for sample in candidates:
            if not isinstance(sample, Mapping):
                raise TypeError("each replay candidate must be a sample mapping")
            record = self._canonical_record(**dict(sample))
            canonical_candidates[record.key] = record

        old_scores = {
            record.key: float(record.selection_score) for record in self._records
        }
        merged_by_key: Dict[Tuple[int, Hashable], DetectionReplayRecord] = {
            record.key: record.clone() for record in self._records
        }
        # Updating an existing key preserves its stable position; new keys use
        # candidate order. Duplicate candidates have already collapsed above.
        merged_by_key.update(canonical_candidates)
        merged = list(merged_by_key.values())
        keep_count = min(target_capacity, len(merged))
        for record in merged:
            record.selection_score = self._score_for(record, scores, score_fn)
            score_was_supplied = score_fn is not None or (
                scores is not None
                and (record.key in scores or record.image_id in scores)
            )
            if score_was_supplied or any(
                key in (record.metadata or {})
                for key in _REFINEMENT_SCORE_METADATA_KEYS
            ):
                if record.metadata is None:
                    record.metadata = {}
                record.metadata[_SELECTION_SCORE_VALID_KEY] = True

        selected_indices = self._utility_refined_indices(
            merged,
            keep_count,
            redundancy_weight=float(feature_redundancy_weight),
        )
        refined = [merged[index] for index in selected_indices]

        old_keys = [record.key for record in self._records]
        new_keys = [record.key for record in refined]
        old_updated = [merged_by_key[key] for key in old_keys]

        resizing = target_capacity != self.current_capacity
        proposed_is_better = (bool(refined) or target_capacity == 0) and (
            resizing
            or self._selection_utility(
                refined, redundancy_weight=float(feature_redundancy_weight)
            )
            >= self._selection_utility(
                old_updated, redundancy_weight=float(feature_redundancy_weight)
            )
        )
        scores_changed = any(
            key in old_scores and old_scores[key] != float(record.selection_score)
            for key, record in zip(new_keys, refined)
        )
        changed = proposed_is_better and (
            resizing or old_keys != new_keys or scores_changed
        )

        new_seen_keys = set(canonical_candidates).difference(self._seen_keys)
        self._seen_keys.update(canonical_candidates)
        self._unique_insertions += len(new_seen_keys)
        if proposed_is_better:
            self.current_capacity = target_capacity
            self._records = refined
        self._rebuild_index()
        return changed

    def _rebuild_index(self) -> None:
        self._sampling_cache.clear()
        self._record_indices = {
            record.key: index for index, record in enumerate(self._records)
        }
        if len(self._record_indices) != len(self._records):
            raise ValueError("replay state contains duplicate image records")

    def records(self) -> List[DetectionReplayRecord]:
        return [record.clone() for record in self._records]

    def has_seen(self, *, domain_id: int, image_id: Any) -> bool:
        """Return whether this physical domain/image already had a reservoir draw."""
        key = (int(domain_id), _normalize_image_id(image_id))
        return key in self._seen_keys

    def _sampling_distribution(
        self,
        domain_class_balanced: bool,
        balance_strength: float,
        max_sample_weight: float,
        class_balance_beta: float,
        max_domain_id_exclusive: Optional[int],
        utility_sampling_strength: float,
    ) -> tuple:
        """Cache only population-dependent weights, never per-draw decisions."""
        key = (
            domain_class_balanced,
            balance_strength,
            max_sample_weight,
            class_balance_beta,
            max_domain_id_exclusive,
            utility_sampling_strength,
        )
        if key in self._sampling_cache:
            return self._sampling_cache[key]
        if max_domain_id_exclusive is not None:
            max_domain_id_exclusive = int(max_domain_id_exclusive)
            if max_domain_id_exclusive < 0:
                raise ValueError("max_domain_id_exclusive must be nonnegative")
            eligible_records = [
                record
                for record in self._records
                if record.domain_id < max_domain_id_exclusive
            ]
        else:
            eligible_records = self._records
        if not eligible_records:
            return eligible_records, None, {}, {}

        weights: Optional[torch.Tensor] = None
        domain_weights: Dict[Any, float] = {}
        class_weights: Dict[Any, float] = {}
        if domain_class_balanced:
            domain_counts = Counter(record.domain_id for record in eligible_records)
            class_counts = Counter(
                class_id for record in eligible_records for class_id in record.classes
            )

            def effective_number_weights(counts: Counter) -> Dict[Any, float]:
                keys = tuple(counts)
                if not keys:
                    return {}
                values = torch.tensor(
                    [counts[key] for key in keys], dtype=torch.float64
                )
                beta = float(class_balance_beta)
                weights = (1.0 - beta) / (
                    1.0 - torch.pow(torch.full_like(values, beta), values)
                )
                weights.div_(weights.mean().clamp_min(torch.finfo(weights.dtype).eps))
                return {
                    key: float(weight) for key, weight in zip(keys, weights.tolist())
                }

            domain_weights = effective_number_weights(domain_counts)
            class_weights = effective_number_weights(class_counts)
            raw_weights = []
            for record in eligible_records:
                domain_weight = domain_weights[record.domain_id]
                class_weight = max(
                    (class_weights[class_id] for class_id in record.classes),
                    default=domain_weight,
                )
                # A multi-object image is promoted by its rarest foreground
                # class or its under-represented domain, without multiplying
                # the two corrections and over-sampling it twice.
                raw_weights.append(max(domain_weight, class_weight))
            weights = torch.tensor(raw_weights, dtype=torch.float64)
            weights.div_(weights.mean().clamp_min(torch.finfo(weights.dtype).eps))
            weights.clamp_(max=float(max_sample_weight))
            weights.mul_(float(balance_strength)).add_(1.0 - float(balance_strength))

        if float(utility_sampling_strength) > 0.0:
            utility_weights = self._utility_sampling_weights(
                eligible_records, float(utility_sampling_strength)
            )
            weights = (
                utility_weights if weights is None else weights.mul(utility_weights)
            )

        result = eligible_records, weights, domain_weights, class_weights
        # Keep this bounded when controller settings change.
        self._sampling_cache.clear()
        self._sampling_cache[key] = result
        return result

    def sample_records(
        self,
        batch_size: int,
        *,
        replacement: bool = False,
        domain_class_balanced: bool = False,
        batch_aware_sampling: bool = False,
        balance_strength: float = 0.5,
        max_sample_weight: float = 3.0,
        class_balance_beta: float = 0.999,
        max_domain_id_exclusive: Optional[int] = None,
        utility_sampling_strength: float = 0.0,
    ) -> List[DetectionReplayRecord]:
        """Sample complete images, optionally balancing each sampled batch.

        Global effective-number weights correct the retained population.
        ``batch_aware_sampling`` additionally discounts classes and domains
        already represented by earlier draws in the same mini-batch.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._records:
            return []
        if not 0.0 <= float(balance_strength) <= 1.0:
            raise ValueError("balance_strength must be in 0..1")
        if (
            not math.isfinite(float(max_sample_weight))
            or float(max_sample_weight) < 1.0
        ):
            raise ValueError("max_sample_weight must be finite and at least one")
        if (
            not math.isfinite(float(class_balance_beta))
            or not 0.0 <= float(class_balance_beta) < 1.0
        ):
            raise ValueError("class_balance_beta must be in [0, 1)")
        if (
            not math.isfinite(float(utility_sampling_strength))
            or not 0.0 <= float(utility_sampling_strength) <= 1.0
        ):
            raise ValueError("utility_sampling_strength must be in [0, 1]")
        eligible_records, weights, domain_weights, class_weights = (
            self._sampling_distribution(
                domain_class_balanced,
                float(balance_strength),
                float(max_sample_weight),
                float(class_balance_beta),
                max_domain_id_exclusive,
                float(utility_sampling_strength),
            )
        )
        if not eligible_records:
            return []

        if batch_aware_sampling and domain_class_balanced:
            count = (
                batch_size if replacement else min(batch_size, len(eligible_records))
            )
            indices = []
            available = list(range(len(eligible_records)))
            batch_domains: Counter = Counter()
            batch_classes: Counter = Counter()
            for _ in range(count):
                candidate_indices = (
                    list(range(len(eligible_records))) if replacement else available
                )
                marginal = []
                for index in candidate_indices:
                    record = eligible_records[index]
                    domain_gain = domain_weights.get(record.domain_id, 1.0) / math.sqrt(
                        1.0 + batch_domains[record.domain_id]
                    )
                    class_gain = max(
                        (
                            class_weights.get(class_id, 1.0)
                            / math.sqrt(1.0 + batch_classes[class_id])
                            for class_id in record.classes
                        ),
                        default=domain_gain,
                    )
                    marginal.append(max(domain_gain, class_gain))
                batch_weights = torch.tensor(marginal, dtype=torch.float64)
                batch_weights.div_(
                    batch_weights.mean().clamp_min(torch.finfo(batch_weights.dtype).eps)
                )
                batch_weights.clamp_(max=float(max_sample_weight))
                batch_weights.mul_(float(balance_strength)).add_(
                    1.0 - float(balance_strength)
                )
                if weights is not None:
                    batch_weights.mul_(weights[candidate_indices])
                local_index = int(
                    torch.multinomial(
                        batch_weights,
                        1,
                        replacement=False,
                        generator=self._generator,
                    ).item()
                )
                record_index = candidate_indices[local_index]
                indices.append(record_index)
                record = eligible_records[record_index]
                batch_domains[record.domain_id] += 1
                batch_classes.update(record.classes)
                if not replacement:
                    available.pop(local_index)
        elif weights is not None:
            count = (
                batch_size if replacement else min(batch_size, len(eligible_records))
            )
            indices = torch.multinomial(
                weights,
                count,
                replacement=replacement,
                generator=self._generator,
            ).tolist()
        elif replacement:
            indices = torch.randint(
                len(eligible_records),
                size=(batch_size,),
                generator=self._generator,
            ).tolist()
        else:
            count = min(batch_size, len(eligible_records))
            indices = torch.randperm(len(eligible_records), generator=self._generator)[
                :count
            ].tolist()
        return [eligible_records[index].clone() for index in indices]

    def materialize_records(
        self,
        records: Sequence[DetectionReplayRecord],
        *,
        image_loader: Optional[Callable[[str], Any]] = None,
        batch_image_loader: Optional[Callable[[Sequence[str]], Sequence[Any]]] = None,
        transform: Optional[
            Callable[[Any, Dict[str, Any]], Tuple[torch.Tensor, Dict[str, Any]]]
        ] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[Dict[str, Any]]:
        """Consume selected record clones, applying fresh transforms on the caller."""
        if not records:
            return None
        images: List[torch.Tensor] = []
        targets: List[Dict[str, Any]] = []
        horizontal_flip_flags: List[bool] = []
        loaded_by_index: Dict[int, Any] = {}
        path_indices = [
            index
            for index, record in enumerate(records)
            if record.image is None and record.image_path is not None
        ]
        if path_indices and batch_image_loader is not None:
            loaded = list(
                batch_image_loader(
                    [str(records[index].image_path) for index in path_indices]
                )
            )
            if len(loaded) != len(path_indices):
                raise ValueError("batch_image_loader returned the wrong image count")
            loaded_by_index = dict(zip(path_indices, loaded))
        for index, record in enumerate(records):
            # sample_records() already returned a deep clone, so this ephemeral
            # record can be transformed directly without cloning its image and
            # target a second time. The retained buffer record stays untouched.
            image = record.image
            if image is None:
                if index in loaded_by_index:
                    image = loaded_by_index[index]
                elif image_loader is None or record.image_path is None:
                    raise ValueError(
                        "path-backed records require an image_loader to sample"
                    )
                else:
                    image = image_loader(record.image_path)
            elif torch.is_tensor(image):
                image = image.detach()
            target = record.target
            if transform is not None:
                image, target = transform(image, target)
            horizontal_flip_flags.append(bool(target.pop("_carl_replay_hflip", False)))
            if not torch.is_tensor(image):
                raise TypeError(
                    "the replay image must be a tensor after the joint transform"
                )
            if image.ndim != 3:
                raise ValueError(
                    "the replay image must have shape [C, H, W] after transform"
                )
            image = image.detach()
            if device is not None:
                device = torch.device(device)
                image = image.to(device=device, non_blocking=True)
                target = _to_device(target, device)
            images.append(image)
            targets.append(target)
        return {
            "images": images,
            "targets": targets,
            "domain_ids": torch.tensor(
                [record.domain_id for record in records], dtype=torch.int64
            ),
            "image_ids": [record.image_id for record in records],
            "horizontal_flip_flags": horizontal_flip_flags,
        }

    def domain_counts(self) -> Dict[int, int]:
        return dict(Counter(record.domain_id for record in self._records))

    def class_image_counts(self) -> Dict[int, int]:
        counts = Counter()
        for record in self._records:
            counts.update(record.classes)
        return {class_id: counts[class_id] for class_id in self.foreground_class_ids}

    def class_object_counts(self) -> Dict[int, int]:
        counts = Counter()
        for record in self._records:
            counts.update(torch.as_tensor(record.target["labels"]).tolist())
        return {class_id: counts[class_id] for class_id in self.foreground_class_ids}

    def object_size_counts(self) -> Dict[str, int]:
        counts = Counter()
        for record in self._records:
            target = record.target
            if "area" in target:
                areas = torch.as_tensor(target["area"], dtype=torch.float32).reshape(-1)
            else:
                boxes = torch.as_tensor(target["boxes"], dtype=torch.float32)
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            for value in areas.tolist():
                if value < 32**2:
                    counts["small"] += 1
                elif value < 96**2:
                    counts["medium"] += 1
                else:
                    counts["large"] += 1
        return {name: counts[name] for name in ("small", "medium", "large")}

    def get_fill_ratio(self) -> float:
        if self.current_capacity == 0:
            return 0.0
        return len(self) / self.current_capacity

    def get_capacity_ratio(self) -> float:
        return self.current_capacity / self.max_capacity

    @staticmethod
    def _record_state(record: DetectionReplayRecord) -> Dict[str, Any]:
        return {
            "image_id": copy.deepcopy(record.image_id),
            "domain_id": record.domain_id,
            "target": _clone_cpu(record.target),
            "image": _clone_cpu(record.image),
            "image_path": record.image_path,
            "classes": tuple(record.classes),
            "size_bins": tuple(record.size_bins),
            "selection_score": record.selection_score,
            "metadata": _clone_cpu(record.metadata or {}),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "max_capacity": self.max_capacity,
            "current_capacity": self.current_capacity,
            "foreground_class_ids": self.foreground_class_ids,
            "unique_insertions": self._unique_insertions,
            "seen_keys": copy.deepcopy(
                tuple(sorted(self._seen_keys, key=lambda key: (key[0], repr(key[1]))))
            ),
            "generator_state": self._generator.get_state().clone(),
            "records": [self._record_state(record) for record in self._records],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if int(state_dict.get("version", -1)) != self.STATE_VERSION:
            raise ValueError("unsupported detection replay state version")
        if int(state_dict["max_capacity"]) != self.max_capacity:
            raise ValueError("checkpoint max_capacity does not match the buffer")
        if tuple(state_dict["foreground_class_ids"]) != self.foreground_class_ids:
            raise ValueError("checkpoint foreground classes do not match")
        current_capacity = int(state_dict["current_capacity"])
        if not 0 <= current_capacity <= self.max_capacity:
            raise ValueError("checkpoint current_capacity is invalid")

        records = []
        for raw in state_dict["records"]:
            record = self._canonical_record(
                image_id=raw["image_id"],
                domain_id=raw["domain_id"],
                target=raw["target"],
                image=raw.get("image"),
                image_path=raw.get("image_path"),
                selection_score=raw.get("selection_score", 0.0),
                metadata=raw.get("metadata", {}),
            )
            records.append(record)
        if len(records) > current_capacity:
            raise ValueError("checkpoint contains more records than its capacity")

        self.current_capacity = current_capacity
        self._records = records
        self._unique_insertions = int(state_dict["unique_insertions"])
        self._seen_keys = {
            (int(domain_id), _normalize_image_id(image_id))
            for domain_id, image_id in state_dict["seen_keys"]
        }
        if any(record.key not in self._seen_keys for record in records):
            raise ValueError("checkpoint replay records are missing from seen_keys")
        if self._unique_insertions != len(self._seen_keys):
            raise ValueError("checkpoint unique insertion count is inconsistent")
        self._rebuild_index()
        self._generator.set_state(
            torch.as_tensor(state_dict["generator_state"], dtype=torch.uint8)
        )

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Any]) -> "DetectionReplayBuffer":
        buffer = cls(
            initial_capacity=int(state_dict["current_capacity"]),
            max_capacity=int(state_dict["max_capacity"]),
            foreground_class_ids=tuple(state_dict["foreground_class_ids"]),
        )
        buffer.load_state_dict(state_dict)
        return buffer

    def __len__(self) -> int:
        return len(self._records)
