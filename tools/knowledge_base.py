# tools/knowledge_base.py
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI


class KnowledgeBaseTool(BaseTool):
    """知识库查询工具"""

    def __init__(self, knowledge_base, category=None):
        super().__init__()
        self.knowledge_base = knowledge_base
        self.category = category
        self.name = f"{category}_knowledge_search" if category else "knowledge_search"
        self.description = "在知识库中搜索信息。提供关键词或问题，返回相关信息。"

    def _run(self, query: str) -> str:
        """运行工具"""
        if not self.knowledge_base:
            return "知识库尚未初始化"

        retriever = self.knowledge_base.as_retriever(
            search_kwargs={"k": 5, "filter": {"category": self.category} if self.category != "general" else None}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=retriever
        )

        result = qa_chain.run(query)
        return result

    async def _arun(self, query: str) -> str:
        """异步运行方法"""
        return self._run(query)
