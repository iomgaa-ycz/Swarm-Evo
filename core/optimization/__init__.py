"""
优化模块

包含用于分析和改进系统性能的组件，特别是prompt反思器、生成器和版本管理器。
"""

from .generator import GenerationResult, PromptGenerator
from .reflector import PerformanceMetrics, PromptReflector
from .version_manager import (
    AgentEvolutionRecord,
    AgentVersionManager,
    NodeMetadata,
    PromptVersionRecord,
    TaskReviewRecord,
)

__all__ = [
    'PromptReflector',
    'PerformanceMetrics',
    'PromptGenerator',
    'GenerationResult',
    'AgentVersionManager',
    'AgentEvolutionRecord',
    'PromptVersionRecord',
    'NodeMetadata',
    'TaskReviewRecord'
]
