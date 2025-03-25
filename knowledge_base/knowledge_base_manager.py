# knowledge_base/knowledge_base_manager.py
from .document_loader import DocumentLoader
import os
from ..infrastructure.config import Config


class KnowledgeBaseManager:
    """知识库管理器"""

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.knowledge_base_path = Config.KNOWLEDGE_BASE_PATH

    def initialize_knowledge_base(self):
        """初始化知识库，加载所有文档"""
        os.makedirs(self.knowledge_base_path, exist_ok=True)

        # 加载产品文档
        product_path = os.path.join(self.knowledge_base_path, "product")
        if os.path.exists(product_path):
            print("加载产品文档...")
            product_docs = DocumentLoader.load_directory(
                product_path,
                metadata={"category": "product"}
            )
            product_chunks = DocumentLoader.split_documents(product_docs)
            self.vector_store.add_documents(product_chunks)

        # 加载技术文档
        technical_path = os.path.join(self.knowledge_base_path, "technical")
        if os.path.exists(technical_path):
            print("加载技术文档...")
            technical_docs = DocumentLoader.load_directory(
                technical_path,
                metadata={"category": "technical"}
            )
            technical_chunks = DocumentLoader.split_documents(technical_docs)
            self.vector_store.add_documents(technical_chunks)

        print("知识库初始化完成")

    def add_document(self, file_path, category=None):
        """添加单个文档到知识库"""
        metadata = {"category": category} if category else {}
        documents = DocumentLoader.load_file(file_path, metadata)
        chunks = DocumentLoader.split_documents(documents)
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def search(self, query, category=None, top_k=3):
        """搜索知识库"""
        filter_dict = {"category": category} if category else None
        return self.vector_store.search(query, filter=filter_dict, top_k=top_k)