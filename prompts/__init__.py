"""
提示模板包，包含各种Agent的系统提示
"""

from prompts.file_parser import SYSTEM_MESSAGE as FILE_PARSER_PROMPT
from prompts.output_formatter import SYSTEM_MESSAGE as OUTPUT_FORMATTER_PROMPT
from prompts.product_expert import SYSTEM_MESSAGE as PRODUCT_EXPERT_PROMPT
from prompts.customer_service import SYSTEM_MESSAGE as CUSTOMER_SERVICE_PROMPT
from prompts.router import get_router_prompt, ROUTER_PROMPT
from prompts.knowledge_base import SYSTEM_MESSAGE as KNOWLEDGE_BASE_PROMPT

__all__ = [
    'FILE_PARSER_PROMPT',
    'get_router_prompt',
    'ROUTER_PROMPT',
    'OUTPUT_FORMATTER_PROMPT',
    'PRODUCT_EXPERT_PROMPT',
    'CUSTOMER_SERVICE_PROMPT',
    'KNOWLEDGE_BASE_PROMPT'
]
