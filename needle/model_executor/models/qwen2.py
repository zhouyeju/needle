from needle.framework import Module
from needle.model_executor.layers.vocab_embedding import VocabEmbedding


class Qwen2(Module):
    def __init__(self, config, prefix="model", backend="cpu"):
        super().__init__()
        self.backend = backend
        self.embed = VocabEmbedding(config.vocab_size, config.hidden_size, prefix="embed_tokens", backend=backend)

    def forward(self, input_ids):
        hidden_states = self.embed(input_ids)
        return hidden_states