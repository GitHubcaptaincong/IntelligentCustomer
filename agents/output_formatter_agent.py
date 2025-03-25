import uuid

from langchain_core.messages import SystemMessage, HumanMessage

from agents.base_agent import BaseAgent
from utils import log_util
from utils.user_info import User


class OutputFormatterAgent(BaseAgent):
    """输出格式化Agent，负责整理和美化最终回复"""

    def __init__(self, llm):
        # 输出格式化Agent不需要特定工具
        tools = []

        # 输出格式化的系统提示
        self.system_message = """你是输出格式化Agent，负责整理和优化最终回复给用户的内容。

你的主要职责包括:
1. 调整回复的格式，使其更易读
2. 确保回复的语气一致且专业友好
3. 补充可能遗漏的重要信息
4. 添加适当的问候和结束语
5. 确保回复简洁明了，重点突出

注意:
- 不要改变回复的实质内容和专业建议
- 保持原有的专业术语和技术准确性
- 添加适当的标点和段落分隔，提高可读性
- 确保回复的语气与企业形象一致
        """

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
                SystemMessage(content=self.system_message),
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
                SystemMessage(content=self.system_message),
                HumanMessage(content=f"请格式化以下回复内容:\n\n{query}")
            ]

            response = await self.arun(messages, user_info)
            formatted_content = response["messages"][-1].content


            return formatted_content
        except Exception as e:
            log_util.log_exception(e)
            raise
