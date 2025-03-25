# agents/base_agent.py
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI
from abc import abstractmethod

from utils.log_util import log_exception
from utils.user_info import User
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


def _create_langfuse_callback(user_info: User):
    """获取Langfuse监控"""
    return CallbackHandler(
        user_id=user_info.user_id,
        session_id=user_info.session_id,
    )


class BaseAgent:
    """基础Agent类，提供通用功能"""

    def __init__(self,
                 llm: ChatOpenAI,
                 name: str,
                 agent_type: str,
                 description: str,
                 tools=None):
        self.llm = llm
        self.name = name
        self.agent_type = agent_type
        self.description = description
        self.tools = tools or []
        self.agent_executor = create_react_agent(self.llm, self.tools, checkpointer=MemorySaver())

    def run(self, messages, user_info: User) -> str:
        """运行Agent"""
        try:
            return self.agent_executor.invoke(input=messages,
                                              config={"callbacks": [_create_langfuse_callback(user_info)],
                                                      "configurable": {"thread_id": user_info.session_id}}
                                              )
        except Exception as e:
            log_exception(e)
            raise

    async def arun(self, messages, user_info: User) -> str:
        """异步运行Agent"""
        try:
            resp = await self.agent_executor.ainvoke(input={"messages":messages},
                                              config={"callbacks": [_create_langfuse_callback(user_info)],
                                                      "configurable": {"thread_id": user_info.session_id}}
                                              )
            return resp
        except Exception as e:
            log_exception(e)
            raise

    def get_agent_info(self) -> dict:
        """获取Agent信息"""
        return {
            "name": self.name,
            "type": self.agent_type,
            "description": self.description,
            "tools": [tool.name for tool in self.tools]
        }

    @abstractmethod
    def process_query(self, query, user_info: User):
        """同步处理用户查询"""
        pass

    @abstractmethod
    async def aprocess_query(self, query, user_info: User):
        """异步处理用户查询"""
        pass

