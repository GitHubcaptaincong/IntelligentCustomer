import logging
import os
import traceback

# 获取日志级别
LOG_LEVEL = os.getenv("LOG_LEVEL", "ERROR").upper()

# 配置日志格式
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.ERROR),
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 创建日志工具
logger = logging.getLogger("AppLogger")

def log_error(message: str, *args):
    """普通错误日志"""
    logger.error(message, *args)

def log_exception(e: Exception, message: str = "发生异常"):
    """处理异常并记录完整的错误堆栈"""
    error_msg = f"{message}: {str(e)}\n{traceback.format_exc()}"
    logger.error(error_msg)
