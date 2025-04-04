from typing import Dict, Any, List, Optional, Tuple, Union
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools import Tool, StructuredTool
from agents.base_agent import BaseAgent
from agents.agent_registry import AgentRegistry
from prompts.router import get_router_prompt
from utils.user_info import User
from knowledge_base.knowledge_base_manager import KnowledgeBaseManager
from knowledge_base.vector_store import VectorStoreFactory
from langchain.schema import Document
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate

import json
import logging
import asyncio
from utils import log_util
from infrastructure.config import Config

logger = logging.getLogger(__name__)


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
        # 初始化知识库
        self.vector_store = VectorStoreFactory.create_vector_store()
        self.kb_manager = KnowledgeBaseManager(self.vector_store)
        
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
        
        # 创建记忆信息工具
        remember_tool = StructuredTool.from_function(
            func=self.remember_user_info_async,
            name="remember_info",
            description="""用于记住用户提供的重要信息，稍后可以在对话中使用。
            
            参数：
            - info: 要记住的信息内容（字符串）
            - user_id: 用户ID
            - info_type: 信息类型，如'preference'（偏好）、'profile'（个人资料）等
            
            返回：确认信息已被记忆
            
            使用示例：
            当用户要求记住他们的偏好、个人信息或其他重要内容时。
            """
        )
        
        super().__init__(
            llm=llm,
            name="router_agent",
            agent_type="router",
            description="负责处理用户对话，并在需要时调用专家Agent",
            tools=[route_tool, remember_tool]
        )

    def process_query(self, query: str, user_info: User) -> str:
        """处理用户查询"""
        try:
            # 获取用户记忆信息
            user_memory = self._get_user_memory(user_info.user_id)
            
            # 创建消息
            messages = self._create_messages(query, user_memory, user_info)
            response = self.run(messages, user_info)
            return response["messages"][-1].content
        except Exception as e:
            logger.error(f"处理查询时出错: {str(e)}", exc_info=True)
            return f"处理请求时出错: {str(e)}"

    async def aprocess_query(self, query: str, user_info: User) -> str:
        """异步处理用户查询"""
        try:
            # 获取用户记忆信息
            user_memory = await self._aget_user_memory(user_info.user_id)
            
            # 创建消息
            messages = self._create_messages(query, user_memory, user_info)
            response = await self.arun(messages, user_info)
            return response["messages"][-1].content
        except Exception as e:
            logger.error(f"异步处理查询时出错: {str(e)}", exc_info=True)
            return f"处理请求时出错: {str(e)}"

    def remember_user_info(self, info: str, user_id: str, info_type: str = "general") -> str:
        """记住用户提供的信息"""
        try:
            # 创建一个包含用户信息的文档
            metadata = {
                "user_id": user_id,
                "info_type": info_type,
                "category": "user_memory"
            }
            
            memory_doc = Document(
                page_content=info,
                metadata=metadata
            )
            
            # 添加到向量存储
            self.vector_store.add_documents([memory_doc])
            
            logger.info(f"已记住用户 {user_id} 的信息：类型 - {info_type}, 内容 - {info[:30]}...")
            return f"我已记住这条信息：{info}"
        except Exception as e:
            logger.error(f"记忆用户信息时出错: {str(e)}", exc_info=True)
            return f"记忆信息时出错: {str(e)}"

    async def remember_user_info_async(self, info: str, user_id: str, info_type: str = "general") -> str:
        """异步记住用户提供的信息"""
        try:
            # 使用线程池异步执行
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.remember_user_info, info, user_id, info_type
            )
            return result
        except Exception as e:
            logger.error(f"异步记忆用户信息时出错: {str(e)}", exc_info=True)
            return f"记忆信息时出错: {str(e)}"

    def _get_user_memory(self, user_id: str) -> str:
        """获取用户的历史记忆信息"""
        try:
            # 获取存储类型
            store_type = Config.VECTOR_STORE_TYPE
            results = None
            
            if store_type == "chroma":
                # Chroma使用的过滤器格式
                filter_dict = {"metadata": {"$and": [
                    {"user_id": {"$eq": user_id}},
                    {"category": {"$eq": "user_memory"}}
                ]}}
                results = self.vector_store.search(
                    "用户信息", filter=filter_dict, top_k=10
                )
            else:
                # FAISS和其他向量存储的过滤器格式
                filter_dict = {"user_id": user_id, "category": "user_memory"}
                results = self.vector_store.search(
                    "用户信息", filter=filter_dict, top_k=10
                )
            
            if not results:
                logger.info(f"未找到用户 {user_id} 的记忆信息")
                return ""
            
            # 整合所有记忆信息
            memories = []
            for doc in results:
                info_type = doc.metadata.get("info_type", "general")
                memories.append(f"- {info_type}: {doc.page_content}")
            
            memory_text = "\n".join(memories)
            logger.info(f"获取到用户 {user_id} 的记忆信息: {len(memories)} 条")
            return memory_text
        except Exception as e:
            logger.error(f"获取用户记忆时出错: {str(e)}", exc_info=True)
            return ""

    async def _aget_user_memory(self, user_id: str) -> str:
        """异步获取用户的历史记忆信息"""
        try:
            # 使用线程池异步执行
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._get_user_memory, user_id
            )
            return result
        except Exception as e:
            logger.error(f"异步获取用户记忆时出错: {str(e)}", exc_info=True)
            return ""

    def _create_messages(self, query: str, user_memory: str = "", user_info: User = None) -> List[Dict[str, Any]]:
        """创建消息列表"""
        # 使用带参数的提示词模板
        router_prompt = get_router_prompt(user_memory)
        
        return [
            SystemMessage(content=router_prompt),
            HumanMessage(content=query)
        ]