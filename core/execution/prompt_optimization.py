"""
Prompt Optimization Methods

提供任务执行流程、成功/失败处理、学习/反思流程等功能
"""

import asyncio
import traceback
from typing import Dict, Any

from core.agent.base_agent import BaseReActAgent
from core.agent.prompt_manager import PromptContext
from core.execution.task_class import Task
from core.execution.task_result import TaskExecutionResult
from core.execution.prompt_construction import get_task_description
from utils.logger_system import log_msg


async def prepare_task_execution(
    self,
    agent: BaseReActAgent,
    task: Task
) -> Dict[str, Any]:
    """
    准备任务执行所需的上下文数据

    Returns:
        包含 prompt_context 和 task_description 的字典
    """

    # 构建Prompt上下文
    from core.execution.prompt_construction import construct_prompt_context
    prompt_context = construct_prompt_context(self, task)
    task_description = get_task_description(self, task)

    return {
        "prompt_context": prompt_context,
        "task_description": task_description
    }


async def execute_agent_task(
    self,
    agent: BaseReActAgent,
    task: Task,
    prepared_data: Dict[str, Any]
) -> TaskExecutionResult:
    """
    执行Agent任务

    Returns:
        TaskExecutionResult: 包含执行结果的封装对象
    """
    from core.execution.task_result import TaskExecutionResult

    task_id = task['id']
    task_type = task['type']

    # 统一确定步数限制
    current_max_steps = agent.max_steps
    if task_type == 'review':
        current_max_steps = 1

    agent_input_state = {
        "task_description": prepared_data['task_description'],
        "prompt_context": prepared_data['prompt_context'],
        "max_steps": current_max_steps
    }

    result = await agent(agent_input_state)

    return TaskExecutionResult(
        success=result.get('agent_success', False),
        agent_name=agent.name,
        task_type=task_type,
        task_id=task_id,
        agent_output=result.get('agent_output', {}),
        raw_session=result.get('raw_session'),
        error=result.get('error')
    )


async def handle_successful_task(
    self,
    agent: BaseReActAgent,
    task: Task,
    execution_result: TaskExecutionResult
):
    """
    处理成功任务的路由器

    根据任务类型分发到具体的处理器
    """
    task_type = task['type']

    # 路由到具体的任务处理器
    if task_type == 'review':
        await process_review_task(self, agent, task, execution_result)
    elif task_type in ['explore', 'merge']:
        await process_explore_merge_task(self, agent, task, execution_result)
    else:
        log_msg("WARNING", f"未知的任务类型: {task_type}")


async def handle_failed_task(
    self,
    agent: BaseReActAgent,
    task: Task,
    execution_result: TaskExecutionResult
):
    """
    处理失败任务

    失败任务仍然记录prompt使用次数，但不创建节点、不归档文件
    """
    task_id = task['id']
    task_type = task['type']

    log_msg("WARNING", f"任务失败: {task_type} (ID: {task_id}) by {agent.name}")

    # explore/merge任务即使失败也要记录使用次数
    if task_type in ['explore', 'merge']:
        await self.version_manager.record_prompt_usage(agent.name, task_type)

    # 不创建节点、不归档文件


async def process_review_task(
    self,
    agent: BaseReActAgent,
    task: Task,
    execution_result: TaskExecutionResult
):
    """
    处理Review任务的成功结果

    逻辑：
    1. 提取review数据
    2. 记录到版本管理器（用于优化）
    3. 准备update_data供Pipeline更新节点
    """
    task_id = task['id']
    target_id = task['payload'].get('target_node_id')
    agent_output = execution_result.agent_output

    if not target_id:
        log_msg("WARNING", f"Review任务 {task_id} 缺少target_node_id")
        return

    target_node = self.journal.get_node(target_id)
    if not target_node:
        log_msg("WARNING", f"Review任务 {task_id} 的目标节点 {target_id} 不存在")
        return

    # 准备更新数据
    execution_result.update_data = {
        "score": agent_output.get('score'),
        "summary": agent_output.get('summary', ""),
        "is_bug": agent_output.get('is_bug', False),
        "agent_success": execution_result.success
    }

    # 记录review结果到版本管理器（核心优化需求）
    reviewed_prompt_type = target_node.action_type  # 'explore' 或 'merge'
    original_agent_name = target_node.metadata.get('agent_name', agent.name)
    prompt_version_id = target_node.metadata.get('prompt_version_id')  # 获取原始prompt版本ID
    original_task_id = target_node.metadata.get('task_id')  # 获取原始explore/merge任务ID

    await self.version_manager.record_review_result(
        agent_name=original_agent_name,
        prompt_type=reviewed_prompt_type,
        task_id=original_task_id,
        node_id=target_id,  # 被review的Node ID
        score=agent_output.get('score'),
        has_submission=agent_output.get('has_csv_submission', False),
        version_id=prompt_version_id  # 传递原始版本ID
    )

    log_msg("INFO", f"已记录review结果: {original_agent_name} - {reviewed_prompt_type} - "
                  f"task_id={original_task_id} - node_id={target_id} - "
                  f"version_id={prompt_version_id} - score={agent_output.get('score')}")


async def process_explore_merge_task(
    self,
    agent: BaseReActAgent,
    task: Task,
    execution_result: TaskExecutionResult
):
    """
    处理Explore/Merge任务的成功结果

    逻辑：
    1. 记录prompt使用次数（无论是否生成节点）
    2. 从执行结果中创建Node
    3. 归档solution文件
    """
    from core.execution.prompt_construction import create_nodes_from_result, archive_solution_files

    task_id = task['id']
    task_type = task['type']

    # 第一步：记录prompt使用次数（无论是否生成节点）
    log_msg("INFO", f"[DEBUG] 开始记录prompt使用次数...")
    await self.version_manager.record_prompt_usage(agent.name, task_type)
    log_msg("INFO", f"[DEBUG] prompt使用次数记录完成")

    # 第二步：创建节点
    execution_result.result_nodes = create_nodes_from_result(
        self, execution_result, task, agent.name
    )

    if not execution_result.result_nodes:
        log_msg("WARNING", f"{task_type}任务 {task_id} 未生成任何节点，但已记录使用次数")
        # 创建空的TaskReviewRecord（不添加节点记录）
        await self.version_manager.record_review_result(
            agent_name=agent.name,
            prompt_type=task_type,
            task_id=task_id,
            node_id=None
        )
        return

    # 第三步：归档文件
    archive_path = archive_solution_files(self, task_id, self.config.mle_bench_workspace_dir)
    if archive_path:
        execution_result.archive_path = archive_path
        for node in execution_result.result_nodes:
            node.archive_path = archive_path

    log_msg("INFO", f"{agent.name} 完成 {task_type} 任务 {task_id}，生成 {len(execution_result.result_nodes)} 个节点")


def complete_task_in_pipeline(self, task: Task, execution_result: TaskExecutionResult):
    """
    完成任务的最后一步：通知Pipeline

    Pipeline将：
    1. 更新任务状态
    2. 添加节点到Journal
    3. 创建后续任务（如review）
    """
    task_id = task['id']
    log_msg("INFO", f"[DEBUG] _complete_task_in_pipeline 开始: {task_id}")

    self.task_pipeline.complete_task(
        task_id=task_id,
        result_nodes=execution_result.result_nodes,
        update_data=execution_result.update_data if execution_result.update_data else None
    )

    log_msg("INFO", f"[DEBUG] _complete_task_in_pipeline 完成: {task_id}")


async def check_and_run_reflection(
    self,
    agent_name: str,
    task_type: str
):
    """
    检查并执行学习或反思-生成流程

    当prompt使用次数达到阈值时：
    1. 如果综合评分 < SCORE_THRESHOLD，触发学习器（向分数最高的agent学习）
    2. 如果综合评分 >= SCORE_THRESHOLD，触发反思-生成流程

    参数:
        agent_name: agent名称
        task_type: 任务类型 ('explore' 或 'merge')
    """
    # 第一阶段: 检查反思器和生成器是否可用
    if not self.reflector or not self.generator:
        return

    # 第二阶段: 获取当前版本并检查是否达到反思条件
    current_version = self.version_manager.get_current_prompt(agent_name, task_type)
    if not current_version:
        log_msg("WARNING", f"无法获取 {agent_name} 的 {task_type} 当前版本")
        return

    if current_version.used_count < self.REFLECTION_TRIGGER_THRESHOLD:
        log_msg("INFO", f"{task_type} prompt使用次数不足: {current_version.used_count} < {self.REFLECTION_TRIGGER_THRESHOLD}")
        return

    log_msg("INFO", f"检查学习/反思流程: {task_type} 任务当前prompt已使用 {current_version.used_count} 次")

    # 第三阶段: 检查综合评分
    composite_score = current_version.composite_score
    log_msg("INFO", f"{agent_name} 的 {task_type} prompt综合评分: {composite_score:.3f} (阈值: {self.SCORE_THRESHOLD})")

    # 根据评分决定学习还是反思
    if composite_score < self.SCORE_THRESHOLD:
        # 评分较低，触发学习器
        log_msg("INFO", f"评分低于阈值，触发学习器: {agent_name} 需要向高分agent学习")
        await run_learning_for_agent(self, agent_name, task_type)
    else:
        # 评分较高，触发反思-生成流程
        log_msg("INFO", f"评分高于阈值，触发反思流程: {agent_name} 进行自我反思和改进")
        await run_reflection_generation(self, agent_name, task_type)


async def run_learning_for_agent(
    self,
    agent_name: str,
    task_type: str
):
    """
    为特定agent执行学习流程
    当agent的综合评分低于阈值时，向分数最高的agent学习prompt

    参数:
        agent_name: 需要学习的agent名称
        task_type: 任务类型 ('explore' 或 'merge')
    """
    if not self.learner:
        log_msg("WARNING", f"学习器不可用，无法为 {agent_name} 执行学习")
        return

    try:
        # 分析学习机会
        candidates = self.learner.analyze_learning_opportunities(
            prompt_type=task_type,
            min_agents=2
        )

        # 筛选出以当前agent为学生的候选对
        student_candidates = [
            c for c in candidates
            if c.student_agent == agent_name
        ]

        if not student_candidates:
            log_msg("INFO", f"未找到适合 {agent_name} 的学习机会（已是最高分，转为自我反思）")
            # 没有更高分的agent可以学习，转为自我反思-生成流程
            await run_reflection_generation(self, agent_name, task_type)
            return

        # 选择最佳候选（分数差距最大的）
        best_candidate = student_candidates[0]

        if not best_candidate:
            log_msg("WARNING", f"无法为 {agent_name} 选择学习候选")
            return

        log_msg("INFO", f"开始学习: {agent_name} (分数={best_candidate.student_score:.3f}) -> "
                      f"{best_candidate.teacher_agent} (分数={best_candidate.teacher_score:.3f}), "
                      f"差距={best_candidate.score_gap:.3f}")

        # 执行学习
        learning_result = await self.learner.execute_learning(candidate=best_candidate)

        if learning_result.success:
            log_msg("INFO", f"Prompt学习成功: {learning_result.reasoning}")
        else:
            log_msg("WARNING", f"Prompt学习失败: {learning_result.error}")

    except Exception as e:
        log_msg("ERROR", f"学习流程执行失败: {e}")
        log_msg("ERROR", traceback.format_exc())


async def run_reflection_generation(
    self,
    agent_name: str,
    task_type: str
):
    """
    为特定agent执行反思-生成流程
    当agent的综合评分高于阈值时，进行自我反思和改进

    参数:
        agent_name: agent名称
        task_type: 任务类型 ('explore' 或 'merge')
    """
    try:
        # 第一阶段: 获取当前版本记录
        current_version = self.version_manager.get_current_prompt(agent_name, task_type)

        if not current_version:
            log_msg("WARNING", f"无法获取 {agent_name} 的 {task_type} 当前版本")
            return

        # 第二阶段: 使用反思器分析版本
        reflection_result = await self.reflector.analyze_version(
            version_record=current_version
        )

        log_msg("INFO", f"反思分析完成: {task_type} - 准确率={reflection_result['metrics']['avg_accuracy']:.2f}, "
                      f"生成率={reflection_result['metrics']['avg_generation_rate']:.2f}, "
                      f"综合评分={reflection_result['metrics']['composite_score']:.3f}")

        # 第三阶段: 更新当前版本的reflection
        await self.version_manager.update_version_reflection(
            agent_name=agent_name,
            version_id=current_version.version_id,
            reflection=reflection_result.get('reflection', {})
        )
        log_msg("INFO", f"已更新版本reflection: {current_version.version_id}")

        # 第四阶段: 使用生成器生成新prompt
        current_prompt = current_version.prompt_content
        generation_result = await self.generator.generate_new_prompt(
            agent_name=agent_name,
            prompt_type=task_type,
            current_prompt=current_prompt,
            reflection=reflection_result.get('reflection', {})
        )

        if not generation_result.success:
            log_msg("ERROR", f"生成新prompt失败: {generation_result.error}")
            return

        log_msg("INFO", f"新prompt生成成功: {task_type} - 版本={generation_result.version}, "
                      f"修改项={len(generation_result.changes_made)}")

        # 第五阶段: 应用新prompt
        applied = await self.generator.apply_new_prompt(
            prompt_type=task_type,
            new_prompt=generation_result.new_prompt,
            version=generation_result.version
        )

        if applied:
            log_msg("INFO", f"新prompt已应用: {task_type} 模板已更新为版本 {generation_result.version}")

            # 在版本管理器中记录新版本
            await self.version_manager.record_prompt_version(
                agent_name=agent_name,
                version_id=generation_result.version,
                prompt_type=task_type,
                prompt_content=generation_result.new_prompt,
                source="generated",
                previous_version_id=current_version.version_id
            )
        else:
            log_msg("ERROR", f"应用新prompt失败: {task_type}")

    except Exception as e:
        log_msg("ERROR", f"反思流程执行失败: {e}")
        log_msg("ERROR", traceback.format_exc())
