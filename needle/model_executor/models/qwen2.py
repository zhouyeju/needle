from needle.module import Module
from needle.model_executor.layers.vocab_embedding import VocabEmbedding


class Qwen2(Module):
    def __init__(self, config, prefix="model", backend="cpu"):
        super().__init__()
        self.backend = backend
        self.embed = VocabEmbedding(config.vocab_size, config.hidden_size, prefix="embed_tokens", backend=backend)