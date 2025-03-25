import uuid

from agents.base_agent import BaseAgent
from tools.knowledge_base import KnowledgeBaseTool
from langchain.agents import Tool

class CustomerServiceAgent(BaseAgent):
    """客户服务Agent，处理售后问题、投诉和建议"""

    def __init__(self, llm, knowledge_base, langfuse_handler=None):
        # 定义客户服务工具
        tools = [
            KnowledgeBaseTool(
                name="service_policy",
                knowledge_base=knowledge_base,
                category="service",
                description="用于查询客户服务政策、退换货流程、保修信息等"
            ),
            Tool(
                name="ticket_creator",
                func=self._create_service_ticket,
                description="创建客服工单，用于记录客户的投诉或复杂问题"
            )
        ]

        # 客户服务的系统提示
        customer_service_prompt = """你是客户服务Agent，负责处理客户的售后问题、投诉和建议。

你的专业领域包括:
1. 退换货处理和流程指导
2. 保修政策解释和服务申请
3. 客户投诉处理和问题解决
4. 客户满意度跟踪和改进
5. 会员服务和特殊需求处理

使用service_policy工具查询服务政策，对于需要人工跟进的复杂问题，使用ticket_creator工具创建工单。确保你的回答:
- 同理心强，理解并尊重客户的感受
- 解决方案明确，提供清晰的下一步操作指导
- 专业负责，确保客户问题得到妥善处理
- 主动跟进，对于复杂问题提供持续支持
        """

        super().__init__(
            llm=llm,
            tools=tools,
            system_message=customer_service_prompt,
            name="customer_service",
            langfuse_handler=langfuse_handler
        )

    def _create_service_ticket(self, problem_description):
        """创建客服工单"""
        # 实际项目中，这里应该连接到工单系统API
        ticket_id = f"TICKET-{uuid.uuid4().hex[:8].upper()}"
        return f"已创建客服工单，工单号: {ticket_id}。我们会尽快处理您的问题。"
