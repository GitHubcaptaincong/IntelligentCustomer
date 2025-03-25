# tools/code_executor.py

from langchain.tools import BaseTool
import sys
from io import StringIO
import traceback


class PythonREPLTool(BaseTool):
    """Python代码执行工具"""

    def __init__(self):
        super().__init__()
        self.name = "python_repl"
        self.description = "执行Python代码并返回结果。提供完整的Python代码片段。"

    def _run(self, code: str) -> str:
        """执行Python代码"""
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

    async def _arun(self, code: str) -> str:
        """异步运行方法"""
        return self._run(code)
