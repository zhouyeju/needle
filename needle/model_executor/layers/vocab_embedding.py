from needle.module import Module
from typing import Any

class VocabEmbedding(Module):
    def __init__(self, vocab_size, embed_size, prefix="", backend="cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.backend = backend

    def weight_loader_cpu(self, prefix, tensor_dict):
        weight_key = f"{prefix}.weight"
        self.weight = tensor_dict.get(weight_key, None)
        if self.weight is None:
            return
        assert self.weight.shape == (self.vocab_size, self.embed_size), "loaded weight shape mismatch"
        print(f"Loaded weights for {weight_key} with shape {self.weight.shape}")
        return

    def forward_cpu(self, input_ids):
        # Example implementation for CPU backend
        import numpy as np
        hidden_states = np.matmul(input_ids, self.weight)
        return hidden_states