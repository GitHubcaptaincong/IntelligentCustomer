from typing import List, Dict, Any
from .base_agent import BaseAgent
from utils.user_info import User
from tools.file_parser import create_file_parser_tool
from langchain.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge_base.knowledge_base_manager import KnowledgeBaseManager
from knowledge_base.vector_store import VectorStoreFactory
from prompts import KNOWLEDGE_BASE_PROMPT
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from FlagEmbedding import FlagReranker
import math
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBaseAgent(BaseAgent):
    """知识库代理，负责管理知识库内容和基于知识库回答问题"""
    
    def __init__(self, llm, kb_manager=None):
        # 初始化知识库管理器如果未提供
        if kb_manager is None:
            vector_store = VectorStoreFactory.create_vector_store()
            kb_manager = KnowledgeBaseManager(vector_store)
        
        self.kb_manager = kb_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            add_start_index=True
        )
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        
        # 创建文件解析工具
        file_parser_tool = create_file_parser_tool(
            knowledge_base=kb_manager.vector_store,
            name="parse_document",
            description="解析文档并加入知识库。支持PDF、Excel、图片等格式。参数：file_path (文件路径), file_type (可选，文件类型)"
        )
        
        # 创建知识库查询工具
        knowledge_search_tool = StructuredTool.from_function(
            func=self.search_knowledge_base,
            name="search_knowledge_base",
            description="在知识库中搜索相关信息。参数：query (查询内容), category (可选，分类)"
        )
        
        # 创建添加单个文档工具
        add_document_tool = StructuredTool.from_function(
            func=self.add_document_to_kb,
            name="add_document",
            description="添加单个文档到知识库。参数：file_path (文件路径), category (可选，分类)"
        )
        
        # 创建批量处理目录工具
        process_directory_tool = StructuredTool.from_function(
            func=self.process_directory,
            name="process_directory",
            description="处理目录下的所有文件并添加到知识库。参数：directory_path (目录路径), category (可选，分类)"
        )
        
        super().__init__(
            llm=llm,
            name="knowledge_base_agent",
            agent_type="expert",
            description="专门负责管理知识库内容和基于知识库回答问题，可以解析添加新文档，查询知识库内容",
            tools=[file_parser_tool, knowledge_search_tool, add_document_tool, process_directory_tool]
        )
    
    def search_knowledge_base(self, query: str, category: str = None, top_k: int = 5) -> str:
        """在知识库中搜索相关信息"""
        try:
            logger.info(f"开始搜索知识库，查询：{query}，分类：{category}")
            
            # 获取初始搜索结果
            results = self.kb_manager.search(query, category=category, top_k=top_k * 2)
            if not results:
                logger.warning(f"未找到相关结果：{query}")
                return "在知识库中未找到相关信息。"
            
            # 提取文本内容
            search_texts = [doc.page_content for doc in results]
            
            # 重排序
            reranked_texts = self._rerank_results(query, search_texts, top_k)
            
            # 构建上下文
            context = "\n\n".join(reranked_texts)
            
            # 使用LLM生成回答
            prompt = f"""基于以下已知信息，简洁和专业的回答用户的问题，不需要在答案中添加编造成分。
            已知内容：{context}
            问题：{query}"""
            
            response = self.llm.invoke(prompt)
            logger.info(f"成功生成回答：{query}")
            return response.content
            
        except Exception as e:
            logger.error(f"搜索知识库时出错: {str(e)}", exc_info=True)
            return f"搜索知识库时出错: {str(e)}"
    
    def _rerank_results(self, query: str, search_list: List[str], k: int) -> List[str]:
        """对搜索结果进行重排序"""
        try:
            score_text = []
            for search in search_list:
                score = self.reranker.compute_score([query, search])
                score_text.append([score, search])
            
            sorted_data = sorted(score_text, key=lambda x: x[0], reverse=True)
            min_k = max(min(len(sorted_data), 3), math.floor(k / 3))
            return list(map(lambda x: x[1], sorted_data[:min_k]))
        except Exception as e:
            logger.error(f"重排序结果时出错: {str(e)}", exc_info=True)
            return search_list[:k]  # 出错时返回原始结果
    
    def add_document_to_kb(self, file_path: str, category: str = None) -> str:
        """添加单个文档到知识库"""
        try:
            logger.info(f"开始处理文档：{file_path}，分类：{category}")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"文件不存在：{file_path}")
                return f"文件不存在：{file_path}"
            
            # 解析文件
            documents = self.kb_manager.document_loader.load_file(file_path)
            if not documents:
                logger.error(f"无法解析文件：{file_path}")
                return f"无法解析文件: {file_path}"
            
            # 分割文档
            chunks = self.text_splitter.split_documents(documents)
            
            # 添加元数据
            for chunk in chunks:
                chunk.metadata.update({
                    "source": file_path,
                    "category": category,
                    "chunk_index": len(chunks)
                })
            
            # 添加到向量存储
            self.kb_manager.vector_store.add_documents(chunks)
            
            logger.info(f"成功添加文档：{file_path}，共 {len(chunks)} 个块")
            return f"成功将文档添加到知识库，共分割为 {len(chunks)} 个内容块。"
        except Exception as e:
            logger.error(f"添加文档到知识库时出错: {str(e)}", exc_info=True)
            return f"添加文档到知识库时出错: {str(e)}"
    
    def process_directory(self, directory_path: str, category: str = None) -> str:
        """处理目录下的所有文件"""
        try:
            logger.info(f"开始处理目录：{directory_path}，分类：{category}")
            
            if not os.path.exists(directory_path):
                logger.error(f"目录不存在：{directory_path}")
                return f"目录不存在：{directory_path}"
            
            all_text = []
            processed_files = []
            failed_files = []
            
            # 创建线程池
            with ThreadPoolExecutor(max_workers=10) as executor:
                # 收集所有任务
                future_to_path = {}
                
                # 遍历目录下的所有文件
                for path in Path(directory_path).rglob('*'):
                    if path.is_file():
                        file_type = self._get_file_type(str(path))
                        if file_type in ["pdf", "text", "excel", "image"]:
                            future = executor.submit(self._process_single_file, str(path), file_type)
                            future_to_path[future] = str(path)
                
                # 等待所有任务完成
                for future in future_to_path:
                    file_path = future_to_path[future]
                    try:
                        text = future.result()
                        if text:
                            processed_files.append(file_path)
                            all_text.append(f"\n--- Content from {file_path} ---\n{text}\n")
                    except Exception as e:
                        logger.error(f"处理文件 {file_path} 时出错: {str(e)}", exc_info=True)
                        failed_files.append((file_path, str(e)))
            
            if not all_text:
                logger.warning(f"目录中没有可处理的文件：{directory_path}")
                return "没有找到可处理的文件。"
            
            # 创建文档并分割
            documents = [Document(page_content=text) for text in all_text]
            chunks = self.text_splitter.split_documents(documents)
            
            # 添加元数据
            for chunk in chunks:
                chunk.metadata.update({
                    "category": category,
                    "processed_files": processed_files
                })
            
            # 添加到向量存储
            self.kb_manager.vector_store.add_documents(chunks)
            
            # 构建结果消息
            result_msg = f"成功处理 {len(processed_files)} 个文件，共分割为 {len(chunks)} 个内容块。"
            if failed_files:
                result_msg += f"\n处理失败的文件：{len(failed_files)} 个"
            
            logger.info(f"目录处理完成：{directory_path}")
            return result_msg
            
        except Exception as e:
            logger.error(f"处理目录时出错: {str(e)}", exc_info=True)
            return f"处理目录时出错: {str(e)}"
    
    def _get_file_type(self, file_path: str) -> str:
        """获取文件类型"""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pdf':
                return 'pdf'
            elif ext in ['.txt', '.md', '.markdown']:
                return 'text'
            elif ext in ['.xlsx', '.xls']:
                return 'excel'
            elif ext in ['.jpg', '.jpeg', '.png']:
                return 'image'
            return ''
        except Exception as e:
            logger.error(f"获取文件类型时出错: {str(e)}", exc_info=True)
            return ''
    
    def _process_single_file(self, file_path: str, file_type: str) -> Optional[str]:
        """处理单个文件"""
        try:
            logger.debug(f"开始处理文件：{file_path}，类型：{file_type}")
            
            if file_type == "pdf":
                return self.kb_manager.document_loader._parse_pdf(file_path)
            elif file_type == "text":
                return self.kb_manager.document_loader._parse_text(file_path)
            elif file_type == "excel":
                return self.kb_manager.document_loader._parse_excel(file_path)
            elif file_type == "image":
                return self.kb_manager.document_loader._parse_image(file_path)
            return None
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}", exc_info=True)
            return None
    
    def process_query(self, query: str, user_info: User) -> str:
        """处理用户查询"""
        try:
            logger.info(f"开始处理用户查询：{query}")
            messages = self._create_messages(query)
            response = self.run(messages, user_info)
            logger.info("查询处理完成")
            return response["messages"][-1].content
        except Exception as e:
            logger.error(f"处理用户查询时出错: {str(e)}", exc_info=True)
            return f"处理查询时出错: {str(e)}"
    
    async def aprocess_query(self, query: str, user_info: User) -> str:
        """异步处理用户查询"""
        try:
            logger.info(f"开始异步处理用户查询：{query}")
            messages = self._create_messages(query)
            response = await self.arun(messages, user_info)
            logger.info("异步查询处理完成")
            return response["messages"][-1].content
        except Exception as e:
            logger.error(f"异步处理用户查询时出错: {str(e)}", exc_info=True)
            return f"处理查询时出错: {str(e)}"
    
    def _create_messages(self, query: str) -> List[Dict[str, Any]]:
        """创建消息列表"""
        return [
            SystemMessage(content=KNOWLEDGE_BASE_PROMPT),
            HumanMessage(content=query)
        ]