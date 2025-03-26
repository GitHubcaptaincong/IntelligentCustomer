# tools/knowledge_base.py
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI


def create_knowledge_base_tool(knowledge_base, category=None):
    """创建知识库查询工具"""
    
    def query_knowledge_base(query: str) -> str:
        """在知识库中搜索信息"""
        if not knowledge_base:
            return "知识库尚未初始化"

        retriever = knowledge_base.as_retriever(
            search_kwargs={"k": 5, "filter": {"category": category} if category != "general" else None}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=retriever
        )

        result = qa_chain.run(query)
        return result
    
    name = f"{category}_knowledge_search" if category else "knowledge_search"
    description = f"在{category if category else ''}知识库中搜索信息。提供关键词或问题，返回相关信息。"
    
    return StructuredTool.from_function(
        func=query_knowledge_base,
        name=name,
        description=description
    )


class KnowledgeBaseTool:
    """知识库查询工具类（为了保持向后兼容）"""
    
    def __init__(self, knowledge_base, category=None, **kwargs):
        """初始化知识库工具"""
        self.tool = create_knowledge_base_tool(knowledge_base, category)
        self.name = self.tool.name
        self.description = self.tool.description
        
    def run(self, query: str) -> str:
        """运行工具"""
        return self.tool.run(query)
        
    async def arun(self, query: str) -> str:
        """异步运行工具"""
        return await self.tool.arun(query)
        
    def __getattr__(self, name):
        """转发未定义的属性到内部工具"""
        return getattr(self.tool, name)
