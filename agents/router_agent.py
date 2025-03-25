from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools import Tool
from agents.base_agent import BaseAgent
from agents.agent_registry import AgentRegistry
from prompts.router import ROUTER_PROMPT
from utils.user_info import User

from utils import log_util


def _get_expert_list() -> str:
    """获取所有专家Agent的列表"""
    agents = AgentRegistry.get_all_agents()
    expert_list = []
    for name, agent in agents.items():
        if agent.agent_type != "router":
            expert_list.append(f"- {name}: {agent.description}")
    return "\n".join(expert_list)


def _route_to_expert(query: str, expert_name: str, user_info: User) -> str:
    """路由到指定的专家Agent"""
    expert_agent = AgentRegistry.get_agent(expert_name)
    if not expert_agent:
        return f"抱歉，找不到名为 {expert_name} 的专家。"
    return expert_agent.process_query(query, user_info)


async def _route_to_expert_async(query: str, expert_name: str, user_info: User) -> str:
    """异步路由到指定的专家Agent"""
    expert_agent = AgentRegistry.get_agent(expert_name)
    if not expert_agent:
        return f"抱歉，找不到名为 {expert_name} 的专家。"
    return await expert_agent.aprocess_query(query, user_info)


class RouterAgent(BaseAgent):
    """路由Agent，负责处理用户对话，并在需要时调用专家Agent"""

    def __init__(self, llm):
        # 创建路由工具
        route_tool = Tool(
            name="consult_expert",
            func=_route_to_expert,
            description=f"""当需要专业知识时，可以使用此工具咨询专家。
            
            可用的专家列表：
            {_get_expert_list()}
            
            参数：
            - query: 要咨询的具体问题
            - expert_name: 要咨询的专家名称（必须从上面的列表中选择）
            - user_info: 用户信息对象
            
            返回：专家的回答
            
            使用示例：
            如果用户询问产品相关问题，可以调用 product_expert；
            如果需要解析文件，可以调用 file_parser_agent；
            如果需要格式化输出，可以调用 output_formatter_agent。
            """
        )
        
        super().__init__(
            llm=llm,
            name="router_agent",
            agent_type="router",
            description="负责处理用户对话，并在需要时调用专家Agent",
            tools=[route_tool]
        )

    def process_query(self, query: str, user_info: User) -> str:
        """处理用户查询"""
        try:
            messages = self._create_messages(query)
            response = self.run(messages, user_info)
            return response["messages"][-1].content
        except Exception as e:
            return f"处理请求时出错: {str(e)}"

    async def aprocess_query(self, query: str, user_info: User) -> str:
        """异步处理用户查询"""
        try:
            messages = self._create_messages(query)
            response = await self.arun(messages, user_info)
            return response["messages"][-1].content
        except Exception as e:
            return f"处理请求时出错: {str(e)}"

    def _create_messages(self, query: str) -> List[Dict[str, Any]]:
        """创建消息列表"""
        return [
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=query)
        ]