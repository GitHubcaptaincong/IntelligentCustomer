# infrastructure/database.py
import sqlite3
import json
from datetime import datetime


class ConversationDB:
    """对话历史数据库"""

    def __init__(self, db_path="conversations.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        """创建必要的表"""
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            user_query TEXT,
            response TEXT,
            timestamp TEXT,
            metadata TEXT
        )
        ''')
        self.conn.commit()

    def save_conversation(self, session_id, user_query, response, metadata=None):
        """保存对话记录"""
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else "{}"

        cursor.execute(
            "INSERT INTO conversations (session_id, user_query, response, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_query, response, timestamp, metadata_json)
        )
        self.conn.commit()

    def get_conversation_history(self, session_id, limit=10):
        """获取特定会话的历史记录"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT user_query, response, timestamp FROM conversations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit)
        )
        return cursor.fetchall()
