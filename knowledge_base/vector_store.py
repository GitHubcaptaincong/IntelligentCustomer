# knowledge_base/vector_store.py
import os
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from infrastructure.config import Config
from infrastructure.embeddings import EmbeddingProvider


class VectorStoreFactory:
    """向量存储工厂"""

    @staticmethod
    def create_vector_store(store_type=None, embedding=None):
        """创建向量存储"""
        store_type = store_type or Config.VECTOR_STORE_TYPE

        if not embedding:
            if hasattr(Config, "EMBEDDING_TYPE") and Config.EMBEDDING_TYPE == "OPENAI":
                embedding = EmbeddingProvider.get_openai_embeddings()
            elif hasattr(Config, "EMBEDDING_TYPE") and Config.EMBEDDING_TYPE == "DASHSCOPE":
                embedding = EmbeddingProvider.get_local_embeddings()

        if embedding is None:
            raise ValueError(f"没有支持的向量模型: {embedding}")

        if store_type == "chroma":
            path = os.path.join(Config.VECTOR_STORE_PATH, "chroma")
            return ChromaVectorStore(path=path, embedding=embedding)
        elif store_type == "faiss":
            path = os.path.join(Config.VECTOR_STORE_PATH, "faiss")
            return FAISSVectorStore(path=path, embedding=embedding)
        else:
            raise ValueError(f"不支持的向量存储类型: {store_type}")


class BaseVectorStore:
    """基础向量存储抽象类"""

    def __init__(self, path, embedding):
        self.path = path
        self.embedding = embedding
        self.vector_store = None

    def add_documents(self, documents):
        """添加文档到向量存储"""
        raise NotImplementedError

    def search(self, query, filter=None, top_k=3):
        """搜索相关文档"""
        raise NotImplementedError


class ChromaVectorStore(BaseVectorStore):
    """基于Chroma的向量存储"""

    def __init__(self, path, embedding):
        super().__init__(path, embedding)
        self._initialize_store()

    def _initialize_store(self):
        """初始化向量存储"""
        os.makedirs(self.path, exist_ok=True)

        if os.path.exists(self.path) and any(os.scandir(self.path)):
            self.vector_store = Chroma(
                persist_directory=self.path,
                embedding_function=self.embedding
            )
        else:
            self.vector_store = Chroma(
                persist_directory=self.path,
                embedding_function=self.embedding
            )
            # 创建时可以添加默认文档

    def add_documents(self, documents):
        """添加文档到向量存储"""
        if not self.vector_store:
            self._initialize_store()

        self.vector_store.add_documents(documents)
        self.vector_store.persist()

    def search(self, query, filter=None, top_k=3):
        """搜索相关文档"""
        if not self.vector_store:
            return []

        # 确保过滤器格式正确
        if filter and isinstance(filter, dict) and "metadata" not in filter:
            # 转换为Chroma接受的过滤器格式
            metadata_filter = {}
            for key, value in filter.items():
                metadata_filter[key] = {"$eq": value}
            filter = {"metadata": metadata_filter}

        return self.vector_store.similarity_search(
            query=query,
            k=top_k,
            filter=filter
        )


class FAISSVectorStore(BaseVectorStore):
    """基于FAISS的向量存储"""

    def __init__(self, path, embedding):
        super().__init__(path, embedding)
        self._initialize_store()
        self.index_file = os.path.join(self.path, "index.faiss")
        self.docstore_file = os.path.join(self.path, "docstore.pickle")

    def _initialize_store(self):
        """初始化向量存储"""
        os.makedirs(self.path, exist_ok=True)

        if os.path.exists(self.index_file) and os.path.exists(self.docstore_file):
            self.vector_store = FAISS.load_local(
                folder_path=self.path,
                embeddings=self.embedding,
                index_name="index"
            )
        else:
            self.vector_store = FAISS.from_documents(
                documents=[],
                embedding=self.embedding
            )

    def add_documents(self, documents):
        """添加文档到向量存储"""
        if not self.vector_store:
            self._initialize_store()

        if not documents:
            return

        self.vector_store.add_documents(documents)
        self.vector_store.save_local(self.path, index_name="index")

    def search(self, query, filter=None, top_k=3):
        """搜索相关文档"""
        if not self.vector_store:
            return []

        # FAISS不支持过滤，这里手动实现
        results = self.vector_store.similarity_search(query=query, k=top_k * 2)

        if filter:
            filtered_results = []
            for doc in results:
                if all(doc.metadata.get(k) == v for k, v in filter.items()):
                    filtered_results.append(doc)
                    if len(filtered_results) >= top_k:
                        break
            return filtered_results[:top_k]

        return results[:top_k]
