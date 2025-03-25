from typing import Dict, Optional
from agents.base_agent import BaseAgent

class AgentRegistry:
    """Agent注册表，用于管理所有Agent实例"""
    
    _instance = None
    _agents: Dict[str, BaseAgent] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def register_agent(cls, name: str, agent: BaseAgent) -> None:
        """注册一个新的Agent"""
        cls._agents[name] = agent
    
    @classmethod
    def get_agent(cls, name: str) -> Optional[BaseAgent]:
        """根据名称获取Agent"""
        return cls._agents.get(name)
    
    @classmethod
    def get_all_agents(cls) -> Dict[str, BaseAgent]:
        """获取所有已注册的Agent"""
        return cls._agents.copy()
    
    @classmethod
    def clear(cls) -> None:
        """清空所有注册的Agent"""
        cls._agents.clear() 