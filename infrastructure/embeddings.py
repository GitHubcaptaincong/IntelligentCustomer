# infrastructure/embeddings.py
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import OpenAIEmbeddings

from infrastructure.config import Config


class EmbeddingProvider:
    """嵌入模型提供者"""

    @staticmethod
    def get_openai_embeddings():
        """获取OpenAI嵌入模型"""
        return OpenAIEmbeddings()

    @staticmethod
    def get_local_embeddings():
        """获取本地嵌入模型"""
        return DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=Config.DASHSCOPE_API_KEY
        )
