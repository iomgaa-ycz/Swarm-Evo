"""
Gene Selection Methods

提供基因选择、基因计划日志记录和基因注册表更新等功能
"""

import traceback
from typing import Dict, Any, Optional

from core.evolution.gene_selector import select_gene_plan
from core.execution.task_class import Task
from utils.logger_system import log_msg


def maybe_compute_gene_plan(self, task: Task) -> Optional[Dict[str, Any]]:
    """Gene Selection Logic"""
    if not self.use_pheromone_gene_selection:
        return None
    log_msg("INFO", "[GENE-SELECT] Using pheromone + max-sim selection")

    try:
        update_gene_registry_from_journal(self)
        # 直接调用 gene_selector
        gene_plan = select_gene_plan(
            journal=self.journal,
            gene_registry=self.gene_registry,
            current_step=self.current_epoch,
        )
        log_gene_plan(self, gene_plan)
        return gene_plan

    except Exception as exc:
        log_msg(
            "WARNING",
            "Pheromone gene selection failed.\n"
            f"Exception: {exc}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        return None


def log_gene_plan(self, gene_plan: Dict[str, Any]) -> None:
    """Log gene plan"""
    parts = []
    labels = [
        ("data", "data_source"),
        ("model", "model_source"),
        ("loss", "loss_source"),
        ("opt", "optimizer_source"),
        ("reg", "regularization_source"),
        ("init", "initialization_source"),
        ("tricks", "tricks_source"),
    ]
    for label, field in labels:
        spec = gene_plan.get(field)
        if isinstance(spec, dict):
            node_id = spec.get("source_node_id", "")
            gene_id = spec.get("gene_id", "")
            display = f"{node_id[:6]}:{gene_id[:6]}"
        else:
            display = "None"
        parts.append(f"{label}={display}")
    log_msg("INFO", "[GENE-PLAN] " + " ".join(parts))


def update_gene_registry_from_journal(self) -> None:
    """Update gene registry from journal"""
    for node_id, node in self.journal.nodes.items():
        if node_id in self._gene_registry_updated_nodes:
            continue
        pheromone = None
        if node.metadata:
            pheromone = node.metadata.get("pheromone_node")
        if pheromone is None:
            continue
        self.gene_registry.update_from_reviewed_node(node)
        self._gene_registry_updated_nodes.add(node_id)
