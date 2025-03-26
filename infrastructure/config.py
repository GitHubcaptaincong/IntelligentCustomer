# infrastructure/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """配置管理"""

    # 模型配置
    VECTOR_DB_TYPE = None
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODULE_URL = os.getenv("MODULE_URL")
    MODEL_TYPE = os.getenv("MODEL_TYPE", "openai")  # openai 或 local
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

    # 对话模型
    CHAT_MODEL_TYPE = os.getenv("CHAT_MODEL_TYPE")
    CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")

    # 向量模型
    EMBEDDING_TYPE = "DASHSCOPE"
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

    # 向量存储配置
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")

    # 知识库配置
    KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "./knowledge_base")

    # Langfuse配置
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")