# tools/ticket_creator.py

import uuid
from langchain.tools import StructuredTool


def create_service_ticket(problem_description: str) -> str:
    """创建客服工单并返回工单号"""
    # 实际项目中，这里应该连接到工单系统API
    ticket_id = f"TICKET-{uuid.uuid4().hex[:8].upper()}"
    return f"已创建客服工单，工单号: {ticket_id}。我们会尽快处理您的问题。"


def create_ticket_creator_tool(name="ticket_creator", description="创建客服工单，用于记录客户的投诉或复杂问题"):
    """创建工单创建工具"""
    return StructuredTool.from_function(
        func=create_service_ticket,
        name=name,
        description=description
    )


class TicketCreatorTool:
    """工单创建工具类（为了保持向后兼容）"""
    
    def __init__(self, name="ticket_creator", description="创建客服工单，用于记录客户的投诉或复杂问题"):
        """初始化工单创建工具"""
        self.tool = create_ticket_creator_tool(name, description)
        self.name = self.tool.name
        self.description = self.tool.description
        
    def run(self, problem_description: str) -> str:
        """运行工具"""
        return self.tool.run(problem_description)
        
    async def arun(self, problem_description: str) -> str:
        """异步运行工具"""
        return await self.tool.arun(problem_description)
        
    def __getattr__(self, name):
        """转发未定义的属性到内部工具"""
        return getattr(self.tool, name) 