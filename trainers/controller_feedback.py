"""Annotation-support weighting for RL feedback, never benchmark reporting."""

import math
from typing import Sequence

from utils.detection_metrics import DomainAP50Summary


def support_aware_ap(
    result: DomainAP50Summary,
    rare_ids: Sequence[int],
    support_scale: float,
) -> tuple[dict[int, float], float, dict[int, dict[int, float]]]:
    """Weight present class/domain cells by n/(n+scale).

    Counts come only from validation ground truth, not predictions or AP.
    Domains are macro-averaged by the caller. Rare feedback pools the selected
    rare-class cells with the same support weights; no present cell is dropped.
    Missing classes (zero annotations) have no defined AP and are excluded.
    """
    if not math.isfinite(support_scale) or support_scale <= 0:
        raise ValueError("controller feedback support scale must be positive")
    domains = {}
    weights = {}
    rare_sum = rare_weight = 0.0
    for domain, values in result.per_domain.items():
        numerator = denominator = 0.0
        weights[int(domain)] = {}
        for class_id, count in values.target_count_per_class.items():
            if count < 0:
                raise ValueError("validation annotation counts cannot be negative")
            if count == 0:
                continue
            ap = values.per_class_ap50.get(class_id)
            if ap is None or not math.isfinite(float(ap)) or not 0 <= ap <= 1:
                raise ValueError("present validation classes require finite AP in [0,1]")
            weight = count / (count + support_scale)
            weights[int(domain)][int(class_id)] = weight
            numerator += weight * float(ap)
            denominator += weight
            if class_id in rare_ids:
                rare_sum += weight * float(ap)
                rare_weight += weight
        domains[int(domain)] = numerator / denominator if denominator else 0.0
    return domains, rare_sum / rare_weight if rare_weight else 0.0, weights
