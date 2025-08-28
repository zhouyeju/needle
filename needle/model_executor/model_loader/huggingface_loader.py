from typing import Any


class HuggingfaceLoader:
    def __init__(self, model_path) -> None:
        self.load_config(model_path)


    def load_config(self, model_path: str) -> None:
        import json
        import os
        config_path = os.path.join(model_path, "config.json")
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {config_path}")