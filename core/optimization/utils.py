"""
优化模块公共工具

提供LLM响应解析、消息构建等公共功能
"""

from typing import Dict, List, Any, Optional
import json
from pathlib import Path
from jinja2 import Template
from langchain_core.messages import SystemMessage, HumanMessage
from utils.logger_system import log_msg


class LLMResponseParser:
    """LLM响应解析器"""

    @staticmethod
    def extract_json_from_response(response_content: str) -> Optional[Dict[str, Any]]:
        """
        从LLM响应中提取JSON数据

        参数:
            response_content: LLM响应内容（预期为纯JSON字符串）

        返回:
            解析后的字典，解析失败返回None

        注意：
        - 现在要求 LLM 直接返回纯 JSON，不需要 markdown 代码块
        - 保留兼容性逻辑：如果仍有代码块，会尝试提取
        """
        import re

        try:
            # 去除前后空白
            trimmed = response_content.strip()

            # 如果响应以 { 开头，直接解析（预期情况）
            if trimmed.startswith('{'):
                return json.loads(trimmed)

            # 兼容旧格式：尝试提取 ```json ... ``` 代码块
            json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
            matches = re.findall(json_pattern, trimmed, re.DOTALL)

            if matches:
                # 尝试第一个匹配的代码块
                for match in matches:
                    try:
                        json_str = match.strip()
                        parsed = json.loads(json_str)
                        # 验证是否包含预期的键
                        if any(key in parsed for key in ['diagnosis', 'suggestions', 'new_prompt']):
                            return parsed
                    except json.JSONDecodeError:
                        continue

            # 尝试找到第一个 { 和最后一个 }
            first_brace = trimmed.find('{')
            last_brace = trimmed.rfind('}')

            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = trimmed[first_brace:last_brace + 1]
                return json.loads(json_str)

        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def format_response_content(response) -> str:
        """
        格式化LLM响应内容为字符串

        参数:
            response: LangChain响应对象

        返回:
            字符串格式的响应内容
        """
        return (
            response.content if isinstance(response.content, str)
            else str(response.content)
        )


class MessageBuilder:
    """LLM消息构建器"""

    @staticmethod
    def build_llm_messages(
        template_content: str,
        template_vars: Dict[str, Any],
        human_message: str
    ) -> List:
        """
        构建标准的LLM消息

        参数:
            template_content: 模板内容
            template_vars: 模板变量
            human_message: 人类消息

        返回:
            包含SystemMessage和HumanMessage的列表
        """
        template = Template(template_content)
        system_message = template.render(**template_vars)

        return [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]


class FileSaver:
    """文件保存工具"""

    @staticmethod
    def save_result_to_json(
        result: Dict[str, Any],
        filename: str,
        storage_dir: str = "workspace/logs",
        result_type: str = "result"
    ) -> bool:
        """
        保存结果到JSON文件

        参数:
            result: 结果字典
            filename: 文件名
            storage_dir: 存储目录
            result_type: 结果类型（用于日志）

        返回:
            是否保存成功
        """
        storage_path = Path(storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)

        filepath = storage_path / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            log_msg("INFO", f"{result_type}结果已保存到: {filepath}")
            return True
        except Exception as e:
            log_msg("ERROR", f"保存{result_type}结果失败: {e}")
            return False
