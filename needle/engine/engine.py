class Engine:
    def __init__(self, model_path: str, backend: str = "cpu"):
        self.model_path = model_path
        self.backend = backend