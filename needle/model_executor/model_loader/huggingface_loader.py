from typing import Any
from .registry import MODEL_REGISTRY


class HuggingfaceLoader:
    def __init__(self, model_path) -> None:
        self.config = self.load_config(model_path)


    def load_config(self, model_path: str) -> dict[str, Any]:
        import json
        import os
        config_path = os.path.join(model_path, "config.json")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {config_path}")
        return config
    
    def get_config(self) -> dict[str, Any]:
        return self.config
        
    def infer_model_class(self) -> Any:
        model_type = self.config.get("model_type", None)
        if model_type is None:
            raise ValueError("Model type not specified in config")
        model_class = MODEL_REGISTRY.get(model_type, None)
        if model_class is None:
            raise ValueError(f"Model class for type '{model_type}' not found in registry")
        return model_class