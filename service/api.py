# service/api.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel
import uvicorn
import tempfile
import os
import uuid

from service.chat_service import ChatService


# 定义请求和响应模型
class ChatRequest(BaseModel):
    query: str
    session_id: str = None


class ChatResponse(BaseModel):
    session_id: str
    response: str


class FileUploadResponse(BaseModel):
    message: str
    file_id: str
    session_id: str


# 创建应用
app = FastAPI(title="智能客服Agent系统API")
chat_service = ChatService()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理聊天请求"""
    try:
        result = await chat_service.process_message(
            user_query=request.query,
            session_id=request.session_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理消息出错: {str(e)}")


@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 10):
    """获取会话历史"""
    try:
        history = chat_service.get_conversation_history(session_id, limit)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录出错: {str(e)}")


@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(None),
    description: str = Form("这是一个上传的文件，请解析它的内容")
):
    """上传并处理文件"""
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # 获取文件扩展名
        file_extension = os.path.splitext(file.filename)[1].lower().lstrip('.')
        
        # 构建处理消息
        session_id = session_id or str(uuid.uuid4())
        file_query = f"{description}。文件路径: {temp_file_path}，文件类型: {file_extension}，文件名: {file.filename}"
        
        # 使用聊天服务处理文件（会自动路由到FileParserAgent）
        result = await chat_service.process_message(
            user_query=file_query,
            session_id=session_id
        )

        # 清理临时文件
        os.unlink(temp_file_path)
        
        return FileUploadResponse(
            message=result["response"],
            file_id=session_id,  # 使用session_id作为文件标识
            session_id=result["session_id"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理出错: {str(e)}")


def start_api():
    """启动API服务"""
    uvicorn.run(app, host="0.0.0.0", port=8000)