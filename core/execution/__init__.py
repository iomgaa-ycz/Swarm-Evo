"""
执行模块 (Execution Module)

包含迭代控制器及其相关功能模块
"""

from core.execution.task_result import TaskExecutionResult
from core.execution.gene_selection import (
    maybe_compute_gene_plan,
    log_gene_plan,
    update_gene_registry_from_journal
)
from core.execution.prompt_optimization import (
    prepare_task_execution,
    execute_agent_task,
    handle_successful_task,
    handle_failed_task,
    process_review_task,
    process_explore_merge_task,
    complete_task_in_pipeline,
    check_and_run_reflection,
    run_learning_for_agent,
    run_reflection_generation
)
from core.execution.prompt_construction import (
    construct_prompt_context,
    get_task_description,
    create_nodes_from_result,
    archive_solution_files
)

__all__ = [
    "TaskExecutionResult",
    # Gene Selection
    "maybe_compute_gene_plan",
    "log_gene_plan",
    "update_gene_registry_from_journal",
    # Prompt Optimization
    "prepare_task_execution",
    "execute_agent_task",
    "handle_successful_task",
    "handle_failed_task",
    "process_review_task",
    "process_explore_merge_task",
    "complete_task_in_pipeline",
    "check_and_run_reflection",
    "run_learning_for_agent",
    "run_reflection_generation",
    # Prompt Construction
    "construct_prompt_context",
    "get_task_description",
    "create_nodes_from_result",
    "archive_solution_files",
]
