# tools/agent_selector.py
from typing import Dict

from langchain.tools import BaseTool
from pydantic import Field


class AgentSelectorTool(BaseTool):
    """专家Agent选择工具"""
    name = "agent_selector"
    description = "根据用户查询选择合适的专家Agent。输入是用户查询和所需的专家类型。"

    expert_agents: Dict = Field(default_factory=dict)

    def _run(self, input_str: str) -> str:
        """运行工具"""
        parts = input_str.split(":", 1)
        if len(parts) != 2:
            return "输入格式错误。请使用 'agent_type: query' 的格式"

        agent_type, query = parts
        agent_type = agent_type.strip().lower()

        if agent_type not in self.expert_agents:
            return f"找不到专家Agent: {agent_type}。可用的专家有: {list(self.expert_agents.keys())}"

        agent = self.expert_agents[agent_type]
        return agent.run(query)

    async def _arun(self, input_str: str) -> str:
        """异步运行工具"""
        parts = input_str.split(":", 1)
        if len(parts) != 2:
            return "输入格式错误。请使用 'agent_type: query' 的格式"

        agent_type, query = parts
        agent_type = agent_type.strip().lower()

        if agent_type not in self.expert_agents:
            return f"找不到专家Agent: {agent_type}。可用的专家有: {list(self.expert_agents.keys())}"

        agent = self.expert_agents[agent_type]
        return await agent.arun(query)

