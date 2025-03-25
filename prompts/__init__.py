"""
提示模板包，包含各种Agent的系统提示
"""

from prompts.file_parser import SYSTEM_MESSAGE as FILE_PARSER_PROMPT
from prompts.output_formatter import SYSTEM_MESSAGE as OUTPUT_FORMATTER_PROMPT
from prompts.product_expert import SYSTEM_MESSAGE as PRODUCT_EXPERT_PROMPT
from prompts.customer_service import SYSTEM_MESSAGE as CUSTOMER_SERVICE_PROMPT
from prompts.router import ROUTER_PROMPT as COMMON_ROUTE_PROMPT
from prompts.tech_support import SYSTEM_MESSAGE as TECH_SUPPORT_PROMPT

__all__ = [
    'FILE_PARSER_PROMPT',
    'COMMON_ROUTE_PROMPT',
    'OUTPUT_FORMATTER_PROMPT',
    'PRODUCT_EXPERT_PROMPT',
    'CUSTOMER_SERVICE_PROMPT',
    'TECH_SUPPORT_PROMPT'
]
