from http.client import responses

from langchain_core.messages import SystemMessage, HumanMessage
from agents.base_agent import BaseAgent, _create_langfuse_callback
from agents.agent_registry import AgentRegistry
from utils import log_util
from utils.user_info import User


def _get_expert_list() -> str:
    """获取所有专家Agent的列表"""
    agents = AgentRegistry.get_all_agents()
    expert_list = []
    for name, agent in agents.items():
        if agent.agent_type != "router":
            expert_list.append(f"- {name}: {agent.description}")
    return "\n".join(expert_list)


class RouterAgent(BaseAgent):
    """路由Agent，负责查询扩展、理解和意图识别，并将任务分发给专家Agent"""

    def __init__(self, llm):
        # 路由Agent的系统提示
        super().__init__(
            llm=llm,
            name="router_agent",
            agent_type="router",
            description="负责理解用户查询并将其分发给合适的专家Agent处理",
        )

    def process_query(self, query, user_info: User):
        """处理用户查询，包括意图识别和查询扩展"""

        try:
            # 使用LLM进行意图识别和查询扩展
            messages = [
                SystemMessage(content="""分析以下用户查询，执行以下任务:
                    1. 提取关键信息和真实意图
                    2. 明确用户可能未表达但隐含的需求
                    3. 总结用户实际想要解决的问题
                    4. 下面是所有专家Agent的名称和功能描述，确定最合适的专家Agent，然后返回专家agent的名称（从以下列表中选择一个）: {expert_list}
                """.format(expert_list=_get_expert_list())),
                HumanMessage(content=query)
            ]

            response_text = self.run(messages, user_info)["messages"][-1].content

            # 从响应中提取专家Agent名称和增强后的查询
            expert_name = None
            for name, agent in AgentRegistry.get_all_agents().items():
                if agent.agent_type != "router" and name.lower() in response_text.lower():
                    expert_name = name
                    break

            if not expert_name:
                return "抱歉，我无法确定最合适的专家来处理您的查询。请尝试更详细地描述您的需求。"

            # 直接调用选定的专家Agent
            expert_agent = AgentRegistry.get_agent(expert_name)
            return expert_agent.process_query(query, user_info)

        except Exception as e:
            log_util.log_exception(e)
            raise

    async def aprocess_query(self, query, user_info: User):
        """异步处理用户查询"""

        try:
            # 使用LLM进行意图识别和查询扩展
            messages = [
                SystemMessage(content="""分析以下用户查询，执行以下任务:
                    1. 提取关键信息和真实意图
                    2. 明确用户可能未表达但隐含的需求
                    3. 总结用户实际想要解决的问题
                    4. 下面是所有专家Agent的名称和功能描述，确定最合适的专家Agent，然后返回专家agent的名称（从以下列表中选择一个）: {expert_list}
                """.format(expert_list=_get_expert_list())),
                HumanMessage(content=query)
            ]
            responses = await self.arun(messages, user_info)
            response_text = responses["messages"][-1].content

            # 从响应中提取专家Agent名称和增强后的查询
            expert_name = None
            for name, agent in AgentRegistry.get_all_agents().items():
                if agent.agent_type != "router" and name.lower() in response_text.lower():
                    expert_name = name
                    break

            if not expert_name:
                return "抱歉，我无法确定最合适的专家来处理您的查询。请尝试更详细地描述您的需求。"

            # 直接调用选定的专家Agent
            expert_agent = AgentRegistry.get_agent(expert_name)
            return await expert_agent.aprocess_query(query, user_info)

        except Exception as e:
            log_util.log_exception(e)
            raise
