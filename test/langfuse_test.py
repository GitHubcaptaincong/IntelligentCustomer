from langchain_openai import ChatOpenAI

from infrastructure.config import Config
from langfuse.callback import CallbackHandler


def _init_langfuse():
    """初始化Langfuse监控"""
    return CallbackHandler(
        user_id=1
    )


def main():
    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key="sk-lGG9OCzNOlh9tnbW9b0d02D3B34e4f3c867eBf6dC453F615",
        base_url="https://free.v36.cm/v1",
    )

    response = model.invoke("你好", config=_init_langfuse())
    print(response)


if __name__ == "__main__":
    main()