"""
任务执行结果的统一封装
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from core.execution.journal import Node


@dataclass
class TaskExecutionResult:
    """
    任务执行结果的统一封装
    """
    success: bool
    agent_name: str
    task_type: str
    task_id: str
    agent_output: Dict[str, Any]
    raw_session: Any = None
    error: Optional[str] = None
    result_nodes: List[Node] = field(default_factory=list)
    update_data: Dict[str, Any] = field(default_factory=dict)
    archive_path: Optional[str] = None
