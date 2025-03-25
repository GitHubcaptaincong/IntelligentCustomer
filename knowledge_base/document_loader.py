# knowledge_base/document_loader.py
import os
from langchain.document_loaders import (
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentLoader:
    """文档加载器"""

    @staticmethod
    def load_directory(directory_path, metadata=None):
        """加载目录中的所有文档"""
        if not os.path.exists(directory_path):
            return []

        docs = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_docs = DocumentLoader.load_file(file_path, metadata)
                    docs.extend(file_docs)
                except Exception as e:
                    print(f"加载文件 {file_path} 出错: {str(e)}")

        return docs

    @staticmethod
    def load_file(file_path, metadata=None):
        """根据文件类型加载单个文件"""
        if not os.path.exists(file_path):
            return []

        file_extension = os.path.splitext(file_path)[1].lower()
        base_metadata = metadata or {}
        file_metadata = {
            **base_metadata,
            "source": file_path
        }

        # 根据文件扩展名选择合适的加载器
        if file_extension == ".txt":
            loader = TextLoader(file_path)
        elif file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".csv":
            loader = CSVLoader(file_path)
        elif file_extension in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            # 默认作为文本加载
            loader = TextLoader(file_path)

        documents = loader.load()

        # 添加元数据
        for doc in documents:
            doc.metadata.update(file_metadata)

        return documents

    @staticmethod
    def split_documents(documents, chunk_size=1000, chunk_overlap=200):
        """将文档分割为小块"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        return splitter.split_documents(documents)