import needle
from transformers import AutoTokenizer

class Engine:
    def __init__(self, model_path: str, backend: str = "cpu"):
        self.model_path = model_path
        self.backend = backend
        self.loader = needle.huggingface_loader(model_path)
        self.model_class = self.loader.infer_model_class()
        self.model: needle.module = self.model_class(self.loader.get_config(), backend=backend)

    def load_model(self, prefix="model"):
        params_dict = self.loader.load_weights()
        self.model.load_weights(prefix="model", params_dict=params_dict)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def forward(self, *args, **kwargs):
        prompt = kwargs.get("prompt", None)
        if prompt is None:
            raise ValueError("Prompt is required for forward pass")
        tokenized = self.tokenizer([prompt], return_tensors="np")
        input_ids = needle.tensor(tokenized["input_ids"], backend=self.backend)
        print(f"input_ids: {input_ids}")
        return self.model(input_ids)