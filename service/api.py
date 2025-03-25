# service/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from .chat_service import ChatService


# 定义请求和响应模型
class ChatRequest(BaseModel):
    query: str
    session_id: str = None


class ChatResponse(BaseModel):
    session_id: str
    response: str


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


def start_api():
    """启动API服务"""
    uvicorn.run(app, host="0.0.0.0", port=8000)