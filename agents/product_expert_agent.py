# agents/product_expert_agent.py
from langchain_core.messages import SystemMessage, HumanMessage
from agents.base_agent import BaseAgent
from tools.knowledge_base import create_knowledge_base_tool
from utils import log_util
from utils.user_info import User
from prompts import PRODUCT_EXPERT_PROMPT


class ProductExpertAgent(BaseAgent):
    """产品专家Agent，处理产品相关咨询"""

    def __init__(self, llm, knowledge_base):
        # 定义产品专家工具
        tools = [
            create_knowledge_base_tool(knowledge_base, category="product")
        ]

        super().__init__(
            llm=llm,
            name="product_expert",
            agent_type="expert",
            description="负责回答关于产品的所有问题，包括特性、规格、价格等信息",
            tools=tools
        )

    def process_query(self, query, user_info: User):
        """处理用户查询"""
        try:
            messages = [
                SystemMessage(content=PRODUCT_EXPERT_PROMPT),
                HumanMessage(content=query)
            ]
            response = self.run(messages, user_info)
            return response["messages"][-1].content
        except Exception as e:
            log_util.log_exception(e)
            raise

    async def aprocess_query(self, query, user_info: User):
        """异步处理用户查询"""
        try:
            messages = [
                SystemMessage(content=PRODUCT_EXPERT_PROMPT),
                HumanMessage(content=query)
            ]
            response = await self.arun(messages, user_info)
            return response["messages"][-1].content
        except Exception as e:
            log_util.log_exception(e)
            raise
