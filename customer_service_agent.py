import uuid

from agents.base_agent import BaseAgent
from tools.knowledge_base import create_knowledge_base_tool
from tools.ticket_creator import create_ticket_creator_tool
from prompts import CUSTOMER_SERVICE_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage

class CustomerServiceAgent(BaseAgent):
    """客户服务Agent，处理售后问题、投诉和建议"""

    def __init__(self, llm, knowledge_base):
        # 定义客户服务工具
        tools = [
            create_knowledge_base_tool(knowledge_base, category="service"),
            create_ticket_creator_tool()
        ]

        super().__init__(
            llm=llm,
            tools=tools,
            name="customer_service",
            agent_type="expert",
            description="负责处理客户的售后问题、投诉和建议",
        )

    def process_query(self, query, user_info):
        """处理用户查询"""
        messages = [
            SystemMessage(content=CUSTOMER_SERVICE_PROMPT),
            HumanMessage(content=query)
        ]
        
        response = self.run(messages, user_info)
        return response["messages"][-1].content
        
    async def aprocess_query(self, query, user_info):
        """异步处理用户查询"""
        messages = [
            SystemMessage(content=CUSTOMER_SERVICE_PROMPT),
            HumanMessage(content=query)
        ]
        
        response = await self.arun(messages, user_info)
        return response["messages"][-1].content
