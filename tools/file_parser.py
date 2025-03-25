# tools/file_parser.py

from langchain.tools import StructuredTool
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF for PDF
import pandas as pd  # for Excel
from PIL import Image
import pytesseract  # for OCR
import os


class FileParserService:
    """文件解析服务"""
    
    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base
        self.supported_formats = {
            "pdf": self._parse_pdf,
            "xlsx": self._parse_excel,
            "xls": self._parse_excel,
            "jpg": self._parse_image,
            "jpeg": self._parse_image,
            "png": self._parse_image
        }
    
    def parse_file(self, file_path: str, file_type: str = None) -> str:
        """解析文件并返回内容描述"""
        try:
            if not file_type and file_path:
                file_type = os.path.splitext(file_path)[1].lower().lstrip('.')
            
            if file_type not in self.supported_formats:
                return f"抱歉，不支持的文件类型: {file_type}。目前支持的格式有: PDF, Excel, JPG, PNG。"
            
            parser_func = self.supported_formats[file_type]
            chunks = parser_func(file_path)
            
            # 存储到向量数据库
            if self.knowledge_base:
                file_id = self.knowledge_base.add_documents(chunks)
                return f"文件已成功解析并存储。文件ID: {file_id}，共提取了 {len(chunks)} 个内容块。"
            else:
                return f"文件已成功解析，共提取了 {len(chunks)} 个内容块，但未存储。"
        except Exception as e:
            return f"解析文件时出错: {str(e)}"

    def _parse_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """解析PDF文件"""
        doc = fitz.open(file_path)
        chunks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            chunks.append({
                "content": text,
                "metadata": {
                    "source": file_path,
                    "page": page_num + 1,
                    "type": "pdf"
                }
            })
        
        doc.close()
        return chunks

    def _parse_excel(self, file_path: str) -> List[Dict[str, Any]]:
        """解析Excel文件"""
        xls = pd.ExcelFile(file_path)
        chunks = []
        
        for sheet_name in xls.sheet_names:
            sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
            text = sheet_df.to_string()
            chunks.append({
                "content": text,
                "metadata": {
                    "source": file_path,
                    "sheet": sheet_name,
                    "type": "excel"
                }
            })
        
        return chunks

    def _parse_image(self, file_path: str) -> List[Dict[str, Any]]:
        """解析图片文件并提取文本"""
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, lang='chi_sim+eng')
        
        return [{
            "content": text,
            "metadata": {
                "source": file_path,
                "type": "image"
            }
        }]


def create_file_parser_tool(knowledge_base=None, name="parse_file", description="解析上传的文件并提取内容，支持PDF、Excel和图片格式"):
    """创建文件解析工具"""
    file_parser_service = FileParserService(knowledge_base)
    
    return StructuredTool.from_function(
        func=file_parser_service.parse_file,
        name=name,
        description=description
    ) 