# agents/tech_support_agent.py
from agents.base_agent import BaseAgent
from tools.knowledge_base import create_knowledge_base_tool
from tools.code_executor import create_python_repl_tool
from prompts import TECH_SUPPORT_PROMPT


class TechSupportAgent(BaseAgent):
    """技术支持Agent，处理技术问题"""


    def __init__(self, llm, knowledge_base):
        # 定义技术支持工具
        tools = [
            create_knowledge_base_tool(knowledge_base, category="technical"),
            create_python_repl_tool(name="python_executor")
        ]

        super().__init__(
            llm=llm,
            tools=tools,
            name="tech_support",
            agent_type="expert",
            description="负责解决客户的技术问题和故障",
        )

    def process_query(self, query, user_info):
        """处理用户查询"""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        messages = [
            SystemMessage(content=TECH_SUPPORT_PROMPT),
            HumanMessage(content=query)
        ]
        
        response = self.run(messages, user_info)
        return response["messages"][-1].content
        
    async def aprocess_query(self, query, user_info):
        """异步处理用户查询"""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        messages = [
            SystemMessage(content=TECH_SUPPORT_PROMPT),
            HumanMessage(content=query)
        ]
        
        response = await self.arun(messages, user_info)
        return response["messages"][-1].content
