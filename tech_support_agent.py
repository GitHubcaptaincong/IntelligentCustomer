# agents/tech_support_agent.py
from agents.base_agent import BaseAgent
from tools.knowledge_base import KnowledgeBaseTool
from tools.code_executor import PythonREPLTool


class TechSupportAgent(BaseAgent):
    """技术支持Agent，处理技术问题"""


    def __init__(self, llm, knowledge_base, langfuse_handler=None):
        # 定义技术支持工具
        tools = [
            KnowledgeBaseTool(
                name="tech_knowledge",
                knowledge_base=knowledge_base,
                category="technical",
                description="用于查询技术文档、故障排除指南等技术支持相关信息"
            ),
            PythonREPLTool(
                name="python_executor",
                description="执行Python代码进行数据分析、计算或其他技术操作"
            )
        ]

        # 技术支持的系统提示
        tech_support_prompt = """你是技术支持Agent，负责解决客户的技术问题和故障。

你的专业领域包括:
1. 产品安装和配置指导
2. 故障诊断和排除
3. 软件更新和升级支持
4. 系统集成和兼容性问题
5. 技术文档解释

使用tech_knowledge工具查询技术知识库，必要时可使用python_executor工具执行代码来解决问题。确保你的回答:
- 清晰明了，提供逐步的解决方案
- 技术准确，使用正确的术语和概念
- 实用可行，优先提供最简单有效的解决方法
- 耐心详细，考虑到用户可能的技术水平差异
        """

        super().__init__(
            llm=llm,
            tools=tools,
            system_message=tech_support_prompt,
            name="tech_support",
            langfuse_handler=langfuse_handler
        )
