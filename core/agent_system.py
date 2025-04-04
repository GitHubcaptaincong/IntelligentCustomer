from agents.agent_registry import AgentRegistry
from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.product_expert_agent import ProductExpertAgent
from agents.router_agent import RouterAgent
from agents.output_formatter_agent import OutputFormatterAgent
from agents.file_parser_agent import FileParserAgent

from agents.customer_service_agent import CustomerServiceAgent
from infrastructure.config import Config
from infrastructure.models import ModelProvider
from knowledge_base.vector_store import VectorStoreFactory
from utils.log_util import log_exception
from utils.user_info import User


class AgentSystem:
    """智能客服Agent系统，整合所有组件"""

    def __init__(self, model_config=None, vector_db_config=None, langfuse_config=None):
        """初始化Agent系统"""
        # 加载配置
        self.model_config = model_config or {}
        self.vector_db_config = vector_db_config or {}
        self.langfuse_config = langfuse_config or {}

        # 初始化组件
        self.knowledge_base = VectorStoreFactory.create_vector_store("chroma")

        # 清空之前的Agent注册表
        AgentRegistry.clear()

        # 创建所有Agent
        self._create_agents()

    def _init_llm(self, model_type=None, model_name=None):
        """初始化LLM模型"""
        model_type = model_type or self.model_config.get("type", Config.MODEL_TYPE)
        model_name = model_name or self.model_config.get("name", Config.MODEL_NAME)
        api_key = self.model_config.get("api_key", Config.OPENAI_API_KEY)
        model_url = self.model_config.get("model_url", Config.MODULE_URL)

        if model_type.lower() == "openai":
            return ModelProvider.get_openai_model(model_name=model_name, api_key=api_key, model_url=model_url)
        else:
            return ModelProvider.get_local_model(model_name=model_name)

    def _create_agents(self):
        """创建所有Agent"""

        common_llm = self._init_llm()
        list = [
            # 创建知识库代理
            KnowledgeBaseAgent(
                llm=common_llm,
                kb_manager=None  # 使用默认的知识库管理器
            ),

            ProductExpertAgent(
                llm=common_llm,
                knowledge_base=self.knowledge_base
            ),

            CustomerServiceAgent(
                llm=common_llm,
                knowledge_base=self.knowledge_base
            ),

            # 创建文件解析Agent
            FileParserAgent(
                llm=common_llm,
                knowledge_base=self.knowledge_base
            ),

            # 创建输出格式化Agent
            OutputFormatterAgent(
                llm=common_llm
            ),

            # 使用本地微调的模型，创建路由Agent
            RouterAgent(
                llm=self._init_llm(Config.CHAT_MODEL_TYPE, Config.CHAT_MODEL_NAME)
            )
        ]

        for agent in list:
            AgentRegistry.register_agent(agent.name, agent)

    async def process_query(self, query, user_id, session_id):
        """处理用户查询"""

        try:
            # 获取路由Agent
            router_agent = AgentRegistry.get_agent("router_agent")
            if not router_agent:
                raise Exception("Router agent not found")

            # 使用路由Agent处理查询
            response = await router_agent.aprocess_query(query, User(user_id, session_id))

            return response
        except Exception as e:
            log_exception(e)
            raise
