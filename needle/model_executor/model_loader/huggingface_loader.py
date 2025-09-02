from typing import Any
from .registry import MODEL_REGISTRY


class JsonConfig:
    def __init__(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            setattr(self, key, value)

    def __getattr__(self, _):
            return None

class HuggingfaceLoader:
    def __init__(self, model_path) -> None:
        self.config = self.load_config(model_path)
        self.model_path = model_path

    def load_config(self, model_path: str) -> JsonConfig:
        import json
        import os
        config_path = os.path.join(model_path, "config.json")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {config_path}")
        return JsonConfig(config)
    
    def get_config(self) -> JsonConfig:
        return self.config
        
    def load_weights(self) -> dict[str, Any]:
        import ml_dtypes
        import safetensors
        import os
        files = os.listdir(self.model_path)
        params_dict = {}
        for file in files:
            if file.endswith(".safetensors"):
                weights_path = os.path.join(self.model_path, file)
                with safetensors.safe_open(weights_path, framework="np") as f:
                    for k in f.keys():
                        params_dict[k] = f.get_tensor(k)
        if params_dict is None:
            raise FileNotFoundError(f"No safetensors files found in {self.model_path}")
        return params_dict
    
    def load_weight_by_name(self, name, framework) -> Any:
        import ml_dtypes
        import safetensors
        import os
        files = os.listdir(self.model_path)
        for file in files:
            if file.endswith(".safetensors"):
                weights_path = os.path.join(self.model_path, file)
                with safetensors.safe_open(weights_path, framework=framework) as f:
                    if name in f.keys():
                        return f.get_tensor(name)
        return None
        
    def infer_model_class(self) -> Any:
        model_arch = self.config.architectures
        if model_arch is None:
            raise ValueError("Model type not specified in config")
        model_class = MODEL_REGISTRY.get(model_arch[0], None)
        if model_class is None:
            raise ValueError(f"Model class for type '{model_arch}' not found in registry")
        return model_class