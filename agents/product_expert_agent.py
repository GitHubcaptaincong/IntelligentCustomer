# agents/product_expert_agent.py
from langchain_core.messages import SystemMessage, HumanMessage
from agents.base_agent import BaseAgent
from tools.knowledge_base import KnowledgeBaseTool
from utils import log_util
from utils.user_info import User


class ProductExpertAgent(BaseAgent):
    """产品专家Agent，处理产品相关咨询"""

    def __init__(self, llm, knowledge_base):
        # 定义产品专家工具
        tools = [
            KnowledgeBaseTool(
                name="product_knowledge",
                knowledge_base=knowledge_base,
                category="product",
                description="用于查询产品相关知识，包括功能、价格、规格等信息"
            )
        ]

        # 产品专家的系统提示
        system_message = """你是产品专家Agent，负责回答关于产品的所有问题。

你的专业领域包括:
1. 产品特性和功能介绍
2. 产品规格和技术参数
3. 产品价格和促销信息
4. 产品对比和推荐
5. 产品使用建议

使用product_knowledge工具来查询产品知识库。确保你的回答:
- 准确无误，不要编造不存在的产品信息
- 客观专业，突出产品优势但不夸大其词
- 适合客户需求，根据客户的具体情况进行个性化推荐
- 清晰易懂，避免过多专业术语
        """

        super().__init__(
            llm=llm,
            name="product_expert",
            agent_type="expert",
            description="负责回答关于产品的所有问题，包括特性、规格、价格等信息",
            tools=tools
        )
        self.system_message = system_message

    def process_query(self, query, user_info: User):
        """处理用户查询"""
        try:
            messages = [
                SystemMessage(content=self.system_message),
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
                SystemMessage(content=self.system_message),
                HumanMessage(content=query)
            ]
            response = await self.arun(messages, user_info)
            return response["messages"][-1].content
        except Exception as e:
            log_util.log_exception(e)
            raise
