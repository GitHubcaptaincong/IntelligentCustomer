# tools/code_executor.py

from langchain.tools import StructuredTool
import sys
from io import StringIO
import traceback


def execute_python_code(code: str) -> str:
    """执行Python代码并返回结果"""
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    try:
        # 创建安全的本地变量环境
        local_vars = {}

        # 使用exec执行代码
        exec(code, {"__builtins__": __builtins__}, local_vars)
        sys.stdout = old_stdout
        output = mystdout.getvalue()

        # 获取最后一个表达式的结果
        last_line = code.strip().split('\n')[-1]
        if not (last_line.startswith('print') or
                last_line.startswith('import') or
                last_line.startswith('from') or
                last_line.startswith('def') or
                last_line.startswith('class') or
                last_line.startswith('#') or
                '=' in last_line):
            try:
                result = eval(last_line, {"__builtins__": __builtins__}, local_vars)
                if result is not None:
                    output += f"\n{result}"
            except:
                pass

        return output or "代码执行成功，无输出。"
    except Exception as e:
        sys.stdout = old_stdout
        return f"代码执行错误: {str(e)}\n{traceback.format_exc()}"


def create_python_repl_tool(name="python_repl", description="执行Python代码并返回结果。提供完整的Python代码片段。"):
    """创建Python代码执行工具"""
    return StructuredTool.from_function(
        func=execute_python_code,
        name=name,
        description=description
    )


class PythonREPLTool:
    """Python代码执行工具类（为了保持向后兼容）"""
    
    def __init__(self, name="python_repl", description="执行Python代码并返回结果。提供完整的Python代码片段。"):
        """初始化Python代码执行工具"""
        self.tool = create_python_repl_tool(name, description)
        self.name = self.tool.name
        self.description = self.tool.description
        
    def run(self, code: str) -> str:
        """运行工具"""
        return self.tool.run(code)
        
    async def arun(self, code: str) -> str:
        """异步运行工具"""
        return await self.tool.arun(code)
        
    def __getattr__(self, name):
        """转发未定义的属性到内部工具"""
        return getattr(self.tool, name)
