from typing import Any, TypedDict


class Task(TypedDict):
    """任务定义"""

    id: str
    type: str  # "explore", "merge", "review"
    priority: int  # 越高越优先
    payload: dict[str, Any]  # 任务上下文 (prompt params, parent nodes etc.)
    status: str  # "pending", "running", "completed", "failed"
    created_at: float
    agent_name: str | None  # 分配给哪个 Agent
    dependencies: dict[str, str] | None  # 依赖的任务 ID，例如 {"gene_plan_source": "task_123"}
