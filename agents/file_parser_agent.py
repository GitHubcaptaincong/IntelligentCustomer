from typing import List, Dict, Any
from .base_agent import BaseAgent
from utils.user_info import User
from tools.file_parser import create_file_parser_tool
from prompts import FILE_PARSER_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage

class FileParserAgent(BaseAgent):
    def __init__(self, llm, knowledge_base=None):
        # 创建文件解析工具
        file_parser_tool = create_file_parser_tool(knowledge_base)
        
        super().__init__(
            llm=llm,
            name="file_parser_agent",
            agent_type="expert",
            description="专门负责解析和处理各种文件格式，包括PDF、Excel和图片，并提取文本内容",
            tools=[file_parser_tool]
        )

    def process_query(self, query, user_info: User):
        """处理用户查询"""
        messages = self._create_messages(query)
        return self.run(messages, user_info)["messages"][-1].content

    async def aprocess_query(self, query, user_info: User):
        """异步处理用户查询"""
        messages = self._create_messages(query)
        response = await self.arun(messages, user_info)
        return response["messages"][-1].content

    def _create_messages(self, query):
        """创建消息列表"""
        return [
            SystemMessage(content=FILE_PARSER_PROMPT),
            HumanMessage(content=query)
        ] 