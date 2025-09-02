import needle
from typing import Any

class VocabEmbedding(needle.module):
    def __init__(self, vocab_size, embed_size, prefix="", dtype="float32", backend="cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.backend = backend
        self.prefix = prefix
        self.dtype = dtype

    def weight_loader_cpu(self, prefix, *args, **kwargs):
        if "params_dict" not in kwargs:
            raise ValueError("params_dict is required to load weights")
        weight_key = f"{prefix}.weight"
        params_dict = kwargs["params_dict"]
        self.weight = needle.tensor(params_dict.get(weight_key, None), dtype=self.dtype, backend=self.backend)
        if self.weight is None:
            return
        print(f"loading weights for {weight_key}")
        assert self.weight.shape == (self.vocab_size, self.embed_size), f"loaded weight shape mismatch, expected ({self.vocab_size}, {self.embed_size}), got {self.weight.shape}"
        return

    def forward(self, input_ids):
        # Example implementation for CPU backend
        assert isinstance(input_ids, needle.tensor), "input_ids is not a Tensor"
        input_ids = needle.expand_dims(input_ids, -1)
        broadcast_input_ids = needle.repeat(input_ids, self.vocab_size, axis=-1)
        embedding = needle.arange(self.vocab_size, backend=self.backend)
        embedding = needle.expand_dims(embedding, 0)
        embedding = needle.expand_dims(embedding, 0)
        embedding = needle.repeat(embedding, broadcast_input_ids.shape[1], axis=1)
        embedding = needle.repeat(embedding, broadcast_input_ids.shape[0], axis=0)
        embedding_matrix = needle.equal(broadcast_input_ids, embedding)
        hidden_states = embedding_matrix @ self.weight
        return hidden_states