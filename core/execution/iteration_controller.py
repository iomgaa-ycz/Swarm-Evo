"""
迭代控制器 (Iteration Controller)

负责管理Agent的执行流程、任务路由和优化触发
"""

import os
import asyncio
import time
import traceback
from typing import Dict, Any

from core.execution.pipeline import Pipeline
from core.agent.agent_pool import AgentPool
from core.agent.base_agent import BaseReActAgent
from core.execution.journal import Journal
from core.execution.task_class import Task
from core.execution.task_result import TaskExecutionResult
from core.evolution.gene_registry import GeneRegistry
from utils.logger_system import log_msg
from core.optimization.reflector import PromptReflector
from core.optimization.generator import PromptGenerator
from core.optimization.version_manager import AgentVersionManager
from core.optimization.prompt_learner import PromptLearner

# 导入拆分的功能模块
from core.execution import gene_selection
from core.execution import prompt_optimization
from core.execution import prompt_construction


class IterationController:
    """
    迭代控制器

    负责管理Agent的执行流程、任务路由和优化触发
    """

    # 类常量
    SCORE_THRESHOLD = 0.65               # 学习/反思阈值
    REFLECTION_TRIGGER_THRESHOLD = 5     # 反思触发使用次数阈值
    TEMPLATE_MAP = {
        'review': 'evaluate_user_prompt.j2',
        'merge': 'merge_user_prompt.j2',
        'explore': 'explore_user_prompt.j2'
    }

    def __init__(
        self,
        agent_pool: AgentPool,
        task_pipeline: Pipeline,
        journal: Journal,
        config: Any,
        competition_description: str = "",
        conda_packages: str = ""

    ):
        self.agent_pool = agent_pool
        self.task_pipeline = task_pipeline
        self.journal = journal
        self.config = config
        self.competition_description = competition_description
        self.conda_packages = conda_packages

        self.current_epoch = 0
        self.start_time = time.time()

        # Gene Selection
        self.use_pheromone_gene_selection = config.use_pheromone_gene_selection
        self.gene_registry = GeneRegistry()
        self._gene_registry_updated_nodes: set = set()

        # 从 agent_pool 获取 llm 和 prompt_manager（假设所有 agent 相同）
        first_agent = next(iter(self.agent_pool.agents.values()), None)
        self.llm = first_agent.llm if first_agent else None
        self.prompt_manager = first_agent.prompt_manager if first_agent else None

        # 初始化版本管理器（传入 prompt_manager 以获取初始模板）
        self.version_manager = AgentVersionManager(
            storage_dir=os.path.join(self.config.mle_bench_workspace_dir, "agent_evolution"),
            prompt_manager=self.prompt_manager
        )

        # 初始化反思器和生成器
        self.reflector = PromptReflector(llm=self.llm, prompt_manager=self.prompt_manager) if self.llm and self.prompt_manager else None
        self.generator = PromptGenerator(
            llm=self.llm,
            prompt_manager=self.prompt_manager,
            version_manager=self.version_manager
        ) if self.llm and self.prompt_manager else None

        # 初始化学习器
        self.learner = PromptLearner(
            version_manager=self.version_manager,
            llm=self.llm,
            prompt_manager=self.prompt_manager,
            learning_threshold=getattr(self.config, 'learning_threshold', 0.1),
            storage_dir=os.path.join(self.config.mle_bench_workspace_dir, "prompt_learning")
        ) if self.llm and self.prompt_manager else None

    async def run_competition(self):
        """主竞争循环"""
        log_msg("INFO", "Starting competition loop...")
        while self.current_epoch < self.config.mle_bench_epoch_limit:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.config.time_limit_seconds:
                log_msg("WARNING", f"已达到时间限制 ({self.config.time_limit_seconds}秒), 停止竞赛循环。当前耗时: {elapsed_time:.2f}秒")
                break

            self.current_epoch += 1
            log_msg("INFO", f"--- Starting Epoch {self.current_epoch} ---")
            await self.run_epoch()

        log_msg("INFO", "Competition loop finished.")

    async def run_epoch(self):
        """
        执行一轮（epoch）

        一轮定义：所有 Agent 各执行一次任务
        任务分配：动态分配，执行完一个后再从 Pipeline 获取下一个任务
        """
        agents = list(self.agent_pool.agents.values())
        log_msg("INFO", f"Epoch {self.current_epoch}: {len(agents)} 个 Agent 轮询执行")

        for idx, agent in enumerate(agents):
            log_msg("INFO", f"Epoch {self.current_epoch}: 准备获取任务 {idx+1}/{len(agents)}")
            task_item = self.task_pipeline.get_task()
            if not task_item:
                log_msg("INFO", f"Epoch {self.current_epoch}: Pipeline 无更多任务，跳过 {agent.name}")
                continue

            task_item['agent_name'] = agent.name
            log_msg("INFO", f"Epoch {self.current_epoch}: {agent.name} ← {task_item['type']} 任务")

            # 串行执行（避免 workspace 冲突）
            log_msg("INFO", f"Epoch {self.current_epoch}: 开始执行 {agent.name} 的任务")
            await self._run_single_task(agent, task_item)
            log_msg("INFO", f"Epoch {self.current_epoch}: {agent.name} 的任务执行完成")

        log_msg("INFO", f"Epoch {self.current_epoch}: 轮询完成")

    async def _run_single_task(self, agent: BaseReActAgent, task: Task):
        """
        执行单个任务的主入口

        流程：
        0. MERGE任务：计算gene_plan
        1. 检查阶段 - explore/merge 任务执行前检查优化条件
        2. 准备阶段 - 构建上下文
        3. 执行阶段 - 调用Agent
        4. 处理阶段 - 根据成功/失败路由到不同的处理器
        5. 完成阶段 - 通知Pipeline并触发后续任务
        """
        task_id = task['id']
        task_type = task['type']
        log_msg("INFO", f"[DEBUG] _run_single_task 开始: {agent.name}, {task_type}, {task_id}")

        try:
            # 第零阶段：MERGE任务计算gene_plan
            if task_type == 'merge':
                payload = task['payload']
                gene_plan = gene_selection.maybe_compute_gene_plan(self, task)
                payload['gene_plan'] = gene_plan
                if gene_plan is None:
                    log_msg("WARNING", f"[MERGE] Task {task_id} running without gene_plan (fallback merge)")

            # 第一阶段：explore/merge 任务执行前检查优化条件
            if task_type in ['explore', 'merge']:
                log_msg("INFO", f"[DEBUG] 检查优化条件...")
                await prompt_optimization.check_and_run_reflection(self, agent.name, task_type)
                log_msg("INFO", f"[DEBUG] 优化条件检查完成")

            # 第二阶段：准备执行上下文
            log_msg("INFO", f"[DEBUG] 准备执行上下文...")
            prepared_data = await prompt_optimization.prepare_task_execution(self, agent, task)
            log_msg("INFO", f"[DEBUG] 上下文准备完成")

            # 第三阶段：执行Agent任务
            log_msg("INFO", f"[DEBUG] 开始执行Agent任务...")
            execution_result = await prompt_optimization.execute_agent_task(self, agent, task, prepared_data)
            log_msg("INFO", f"[DEBUG] Agent任务执行完成: success={execution_result.success}")

            # 第四阶段：根据执行结果路由处理
            if execution_result.success:
                log_msg("INFO", f"[DEBUG] 处理成功任务...")
                await prompt_optimization.handle_successful_task(self, agent, task, execution_result)
                log_msg("INFO", f"[DEBUG] 成功任务处理完成")
            else:
                log_msg("INFO", f"[DEBUG] 处理失败任务...")
                await prompt_optimization.handle_failed_task(self, agent, task, execution_result)
                log_msg("INFO", f"[DEBUG] 失败任务处理完成")

            # 第五阶段：完成任务（通知Pipeline）
            log_msg("INFO", f"[DEBUG] 完成Pipeline任务...")
            prompt_optimization.complete_task_in_pipeline(self, task, execution_result)
            log_msg("INFO", f"[DEBUG] Pipeline任务完成")

            log_msg("INFO", f"[DEBUG] _run_single_task 完全结束")

        except Exception as e:
            log_msg("ERROR", f"Agent {agent.name} 执行任务 {task_id} 时发生异常: {e}")
            log_msg("ERROR", traceback.format_exc())

            # 标记任务失败
            self.task_pipeline.complete_task(task_id, result_nodes=[], update_data=None)
