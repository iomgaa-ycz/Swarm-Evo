"""Pheromone utilities for node-level statistics."""

from __future__ import annotations

import math
from typing import Optional

from core.execution.journal import Node


def ensure_node_stats(node: Node) -> None:
    """Ensure the node metadata contains pheromone bookkeeping fields."""
    if node.metadata is None:
        node.metadata = {}

    metadata = node.metadata
    usage = metadata.get("usage_count")
    if not isinstance(usage, int):
        metadata["usage_count"] = 0
    success = metadata.get("success_count")
    if not isinstance(success, int):
        metadata["success_count"] = 0
    # Explicit None default helps distinguish between initialized and computed values
    if "pheromone_node" not in metadata:
        metadata["pheromone_node"] = None


def _normalize_score(node: Node, score_min: Optional[float], score_max: Optional[float]) -> float:
    if node.score is None or node.is_buggy:
        return 0.0
    if score_min is None or score_max is None:
        return 0.0
    score_range = score_max - score_min
    if score_range <= 0:
        return 0.0
    return (node.score - score_min) / score_range


def compute_node_pheromone(
    node: Node,
    current_step: int,
    score_min: Optional[float],
    score_max: Optional[float],
    alpha: float = 0.5,
    beta: float = 0.3,
    delta: float = 0.2,
    lambda_: float = 0.05,
) -> float:
    """Compute pheromone for a node without mutating it."""
    norm_score = _normalize_score(node, score_min, score_max)

    usage_count = 0
    success_count = 0
    if node.metadata:
        usage_count = int(node.metadata["usage_count"])
        success_count = int(node.metadata["success_count"])
    usage_denom = max(1, usage_count)
    success_ratio = success_count / usage_denom

    node_step = int(node.metadata["node_step"])  # node-level step for recency
    step_diff = current_step - node_step
    if step_diff < 0:
        step_diff = 0  # 避免偶发顺序问题
    recency = math.exp(-lambda_ * step_diff)


    pheromone = alpha * norm_score + beta * success_ratio + delta * recency
    return float(pheromone)
