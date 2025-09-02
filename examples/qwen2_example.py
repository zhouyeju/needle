import needle
import json


if __name__ == "__main__":
    engine = needle.engine(model_path="/Users/yejuzhou/repo/Qwen2.5-0.5B-Instruct", backend="cpu")
    engine.load_model()
    prompts = json.load(open("prompts", "r"))
    for idx, prompt in enumerate(prompts):
        hidden_states = engine.forward(prompt=prompt)
        hidden_states.numpy().tofile(f"qwen2_prompt_{idx}_test.bin")
