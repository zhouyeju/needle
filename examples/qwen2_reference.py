from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from transformers import AutoTokenizer
import needle
import json
import torch

if __name__ == "__main__":
    model_path = "/Users/yejuzhou/repo/Qwen2.5-0.5B-Instruct"
    huggingface_loader = needle.huggingface_loader(model_path)
    config = huggingface_loader.get_config()
    vocab_size = config.vocab_size # Example vocabulary size
    embedding_dim = config.hidden_size  # Example embedding dimension
    embedding_layer = VocabParallelEmbedding(vocab_size, embedding_dim, params_dtype=torch.float32)
    embedding_weight = huggingface_loader.load_weight_by_name("model.embed_tokens.weight", framework="pt")
    embedding_layer.weight.copy_(embedding_weight)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompts = json.load(open("prompts", "r"))
    for idx, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        embed = embedding_layer(input_ids)
        embed_numpy = embed.detach().cpu().numpy()
        embed_numpy.tofile(f"qwen2_prompt_{idx}_golden.bin")
