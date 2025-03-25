class User:
    def __init__(self, user_id: int, session_id: str):
        """初始化用户信息"""
        self.user_id = user_id
        self.session_id = session_id