from needle.framework import Module, Tensor
from typing import Any

class VocabEmbedding(Module):
    def __init__(self, vocab_size, embed_size, prefix="", backend="cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.backend = backend

    def weight_loader_cpu(self, prefix, tensor_dict):
        weight_key = f"{prefix}.weight"
        self.weight = Tensor(tensor_dict.get(weight_key, None), backend=self.backend)
        if self.weight is None:
            return
        assert self.weight.shape == (self.vocab_size, self.embed_size), "loaded weight shape mismatch"
        print(f"Loaded weights for {weight_key} with shape {self.weight.shape}")
        return

    def forward(self, input_ids):
        # Example implementation for CPU backend
        assert isinstance(input_ids, Tensor), "input_ids is not a Tensor"
        hidden_states = input_ids @ self.weight
        return hidden_states