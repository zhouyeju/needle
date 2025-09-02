from needle.model_executor.model_loader.huggingface_loader import HuggingfaceLoader

class Engine:
    def __init__(self, model_path: str, backend: str = "cpu"):
        self.model_path = model_path
        self.backend = backend
        self.loader = HuggingfaceLoader(model_path)
        self.model_class = self.loader.infer_model_class()
        self.model = self.model_class(self.loader.get_config(), backend=backend)

    def load_model(self, prefix="model"):
        self.model.load_weights(prefix="model", model_path=self.model_path)