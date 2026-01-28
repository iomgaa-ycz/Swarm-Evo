"""
Prompt Construction Methods

提供Prompt上下文构建、节点创建和文件归档等功能
"""

import os
import json
import zipfile
import time
import traceback
from typing import List, Dict, Any, Optional

from core.agent.prompt_manager import PromptContext
from core.execution.journal import Node
from core.execution.task_class import Task
from core.execution.task_result import TaskExecutionResult
from utils.logger_system import log_msg


def construct_prompt_context(
    self,
    task: Task,
    step_limit: Optional[int] = None
) -> PromptContext:
    """
    构建Prompt上下文

    从task payload和全局配置中提取数据，构建PromptContext对象

    参数:
        task: 任务字典
        step_limit: 步数限制（已根据任务类型确定）
    """
    payload = task.get('payload', {})

    elapsed = time.time() - self.start_time
    remaining = self.config.time_limit_seconds - elapsed

    # 设置默认模板名称
    if payload.get('template_name') is None:
        payload['template_name'] = self.TEMPLATE_MAP.get(task['type'], 'explore_user_prompt.j2')

        # Explore任务继承处理
        if task['type'] == 'explore':
            parent_id = payload.get('parent_id')
            if parent_id:
                parent_node = self.journal.get_node(parent_id)
                if parent_node:
                    payload['parent_code'] = parent_node.code
                    payload['parent_feedback'] = parent_node.summary
                    # logs 对应模板中的 parent_history
                    payload['parent_history'] = parent_node.logs
                    payload['parent_score'] = parent_node.score

    # 动态填充数据
    candidates_data = {}
    gene_plan_data = {}
    solution_code = None
    execution_logs = None

    if task['type'] == 'merge':
        # Merge 任务逻辑更新
        gene_plan_data = payload.get('gene_plan')
        # 仍然需要 candidate code 用于 materialization
        candidate_ids = payload.get('candidate_ids', [])
        # 如果 gene_plan 存在，sources 也应该作为 candidates
        if gene_plan_data:
            for spec in gene_plan_data.values():
                if isinstance(spec, dict) and spec.get("source_node_id"):
                     candidate_ids.append(spec.get("source_node_id"))

        candidate_ids = list(set(candidate_ids))
        for cid in candidate_ids:
            node = self.journal.get_node(cid)
            if node and node.code:
                candidates_data[cid] = node.code

    elif task['type'] == 'review':
        # 获取被review节点的代码和日志
        target_id = payload.get('target_node_id')
        if target_id:
            node = self.journal.get_node(target_id)
            if node:
                solution_code = node.code
                execution_logs = node.logs

    # 如果没有提供step_limit，使用默认值
    if step_limit is None:
        step_limit = 10  # 默认步数限制

    return PromptContext(
        workspace_root=self.config.mle_bench_workspace_dir,
        conda_env_name=self.config.conda_env_name,
        time_limit_seconds=self.config.time_limit_seconds,
        total_iterations=self.config.mle_bench_epoch_limit,
        iteration=self.current_epoch,
        elapsed_seconds=elapsed,
        remaining_seconds=remaining,
        conda_packages=self.conda_packages,
        task_description=get_task_description(self, task),
        step_limit=step_limit,
        parent_code=payload.get('parent_code'),
        parent_feedback=payload.get('parent_feedback'),
        parent_score=payload.get('parent_score'),
        candidates=candidates_data if candidates_data else payload.get('candidates'),
        gene_plan=gene_plan_data if gene_plan_data else payload.get('gene_plan'),
        solution_code=solution_code if solution_code else payload.get('solution_code'),
        execution_logs=execution_logs if execution_logs else payload.get('execution_logs'),
        parent_history=payload.get('parent_history'),
        template_name=payload.get('template_name')
    )


def get_task_description(self, task: Task) -> str:
    """
    生成任务描述

    基于任务类型生成人类可读的任务描述，并添加竞赛背景
    """
    t_type = task['type']

    task_instructions = {
        'explore': "Please explore a new solution based on the plan.",
        'merge': "Please merge the selected strategies into a new solution.",
        'review': "Please review the solution and provide feedback."
    }

    task_instruction = task_instructions.get(t_type, f"Execute task of type {t_type}")

    # 添加竞赛背景
    if self.competition_description:
        return f"# Competition Background\n{self.competition_description}\n\n---\n\n# Your Task\n{task_instruction}"

    return task_instruction


def create_nodes_from_result(
    self,
    execution_result: TaskExecutionResult,
    task: Task,
    agent_name: str
) -> List[Node]:
    """
    从执行结果中创建Journal节点

    策略：
    1. 从history中提取所有solution.py版本（针对explore/merge）
    2. 如果没有找到，使用final agent output作为fallback

    参数:
        execution_result: 任务执行结果
        task: 任务对象
        agent_name: 执行任务的agent名称

    返回:
        创建的节点列表
    """
    nodes = []
    raw_session = execution_result.raw_session
    task_type = task['type']

    # 处理父节点ID
    parent_ids = []
    if task['type'] == 'merge':
        # Merge node parents = Gene Sources
        gene_plan = task['payload'].get('gene_plan') or {}
        gene_source_ids = [
            spec.get("source_node_id")
            for spec in gene_plan.values()
            if isinstance(spec, dict) and spec.get("source_node_id")
        ]
        # Combine candidate_ids and gene_source_ids
        candidate_ids = task['payload'].get('candidate_ids', [])
        parent_ids = list(set(candidate_ids + gene_source_ids))
    else:
        parent_id = task.get('payload', {}).get('parent_id')
        if parent_id:
            parent_ids = [parent_id]

    # 提取完整日志
    logs = ""
    if raw_session:
        logs = json.dumps([h.get('observation') for h in raw_session.history], ensure_ascii=False)

    # 策略1: 从history中提取solution.py的多个版本
    if task['type'] in ['explore', 'merge'] and raw_session:
        history = raw_session.history
        seen_content = set()

        for i, step in enumerate(history):
            action = step.get('action') or step.get('tool')
            tool_input = step.get('input') or step.get('tool_input', {})

            if action == 'write_file' and isinstance(tool_input, dict):
                path = tool_input.get('path', '')
                content = tool_input.get('content', '')

                if path.endswith('solution.py') and content and content not in seen_content:
                    seen_content.add(content)

                    # 获取当前prompt版本ID（用于review时匹配正确的版本）
                    current_prompt = self.version_manager.get_current_prompt(agent_name, task_type)
                    current_version_id = current_prompt.version_id if current_prompt else None

                    # 查找后续的执行日志
                    execution_log = ""
                    for j in range(i + 1, len(history)):
                        next_step = history[j]
                        next_action = next_step.get('action') or next_step.get('tool')
                        next_input = next_step.get('input') or next_step.get('tool_input', {})

                        if next_action in ['run_python', 'python', 'bash', 'cmd_line', 'execute_script', 'terminal']:
                            execution_log = next_step.get('observation', "")
                            break

                        if next_action == 'write_file' and isinstance(next_input, dict):
                            next_path = next_input.get('path', '')
                            if next_path.endswith('solution.py'):
                                break

                    # 创建节点
                    node = Node(
                        parent_ids=parent_ids,
                        code=content,
                        score=None,
                        step=self.current_epoch,
                        action_type=task['type'],
                        logs=execution_log if execution_log else logs,
                        metadata={
                            "agent_name": execution_result.agent_name,
                            "task_id": task['id'],
                            "success": execution_result.success,
                            "version": "history_snapshot",
                            "prompt_version_id": current_version_id  # 保存当前prompt版本ID
                        }
                    )
                    nodes.append(node)

    # 策略2: Fallback到final agent output
    if not nodes:
        agent_output = execution_result.agent_output
        code_content = ""

        if isinstance(agent_output, dict):
            code_content = agent_output.get('code', "")

        if code_content:
            # 获取当前prompt版本ID
            current_prompt = self.version_manager.get_current_prompt(agent_name, task_type)
            current_version_id = current_prompt.version_id if current_prompt else None

            node = Node(
                parent_ids=parent_ids,
                code=code_content,
                score=None if not isinstance(agent_output, dict) else agent_output.get('score'),
                step=self.current_epoch,
                action_type=task['type'],
                logs=logs,
                metadata={
                    "agent_name": execution_result.agent_name,
                    "task_id": task['id'],
                    "success": execution_result.success,
                    "version": "final_output",
                    "prompt_version_id": current_version_id  # 保存当前prompt版本ID
                }
            )
            nodes.append(node)

    return nodes


def archive_solution_files(
    self,
    task_id: str,
    workspace_dir: str
) -> Optional[str]:
    """
    归档任务执行过程中的 solution.py 和 submission.csv 文件

    参数:
        task_id: 任务ID
        workspace_dir: 工作目录路径

    返回:
        归档文件的路径，如果文件不存在则返回None
    """
    try:
        solution_path = os.path.join(workspace_dir, "solution.py")
        submission_path = os.path.join(workspace_dir, "submission", "submission.csv")

        solution_exists = os.path.exists(solution_path)
        submission_exists = os.path.exists(submission_path)

        if not solution_exists and not submission_exists:
            log_msg("WARNING", f"任务 {task_id}: 未找到 solution.py 或 submission.csv，跳过归档")
            return None

        # 使用 .archives 隐藏目录
        archive_dir = os.path.join(workspace_dir, ".archives")
        os.makedirs(archive_dir, exist_ok=True)

        # 创建归档文件
        archive_filename = f"{task_id}.zip"
        archive_path = os.path.join(archive_dir, archive_filename)

        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if solution_exists:
                zipf.write(solution_path, "solution.py")
                log_msg("INFO", f"已归档 solution.py 到 {archive_path}")

            if submission_exists:
                zipf.write(submission_path, "submission.csv")
                log_msg("INFO", f"已归档 submission.csv 到 {archive_path}")

        return archive_path

    except Exception as e:
        log_msg("ERROR", f"归档文件失败 (任务 {task_id}): {e}")
        log_msg("ERROR", traceback.format_exc())
        return None
