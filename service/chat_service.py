# service/chat_service.py
import uuid

from core.agent_system import AgentSystem
from ..infrastructure.database import ConversationDB


class ChatService:
    """聊天服务，处理用户交互"""

    def __init__(self):
        self.agent_system = AgentSystem()
        self.conversation_db = ConversationDB()

    async def process_message(self, user_query, user_id=None, session_id=None):
        """处理用户消息"""
        session_id = session_id or str(uuid.uuid4())
        user_id = user_id or str(uuid.uuid4())

        # 调用Agent系统处理消息
        response = await self.agent_system.process_query(user_query, user_id, session_id)

        # 保存对话记录
        self.conversation_db.save_conversation(
            session_id=session_id,
            user_query=user_query,
            response=response
        )

        return {"session_id": session_id, "response": response}


    def get_conversation_history(self, session_id, limit=10):
        """获取会话历史"""
        return self.conversation_db.get_conversation_history(session_id, limit)