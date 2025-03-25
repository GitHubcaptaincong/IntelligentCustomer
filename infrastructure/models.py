# infrastructure/models.py
from langchain_openai import ChatOpenAI
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModel, pipeline


class ModelProvider:
    """模型提供者，管理不同模型的加载与使用"""

    @staticmethod
    def get_openai_model(model_name: str, api_key, model_url, temperature: float = 0.2):
        """获取OpenAI模型"""
        return ChatOpenAI(
            base_url=model_url,
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

    @staticmethod
    def get_local_model(model_name="THUDM/chatglm3-6b", device="cuda" if torch.cuda.is_available() else "cpu"):
        """获取本地模型实例"""
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

        return HuggingFacePipeline(pipeline=pipe)

