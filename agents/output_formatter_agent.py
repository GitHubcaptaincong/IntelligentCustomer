import uuid

from langchain_core.messages import SystemMessage, HumanMessage

from agents.base_agent import BaseAgent
from utils import log_util
from utils.user_info import User
from prompts import OUTPUT_FORMATTER_PROMPT


class OutputFormatterAgent(BaseAgent):
    """输出格式化Agent，负责整理和美化最终回复"""

    def __init__(self, llm):
        # 输出格式化Agent不需要特定工具
        tools = []

        super().__init__(
            llm=llm,
            tools=tools,
            name="output_formatter",
            agent_type="formatter",
            description="输出格式化Agent，负责整理和美化最终回复",
        )

    def process_query(self, query, user_info: User):
        """格式化回复内容"""

        try:
            # 使用LLM进行格式化
            messages = [
                SystemMessage(content=OUTPUT_FORMATTER_PROMPT),
                HumanMessage(content=f"请格式化以下回复内容:\n\n{query}")
            ]

            response = self.run(messages, user_info)
            formatted_content = response.content

            return formatted_content
        except Exception as e:
            log_util.log_exception(e)
            raise

    async def aprocess_query(self, query, user_info: User):
        """异步格式化回复内容"""

        try:
            # 使用LLM进行格式化
            messages = [
                SystemMessage(content=OUTPUT_FORMATTER_PROMPT),
                HumanMessage(content=f"请格式化以下回复内容:\n\n{query}")
            ]

            response = await self.arun(messages, user_info)
            formatted_content = response["messages"][-1].content

            return formatted_content
        except Exception as e:
            log_util.log_exception(e)
            raise
