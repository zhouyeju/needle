from needle import Engine


if __name__ == "__main__":
    engine = Engine(model_path="path/to/model", backend="cpu")
    # Add code to use the engine for inference or other tasks
    print(f"Engine initialized with model path: {engine.model_path} and backend: {engine.backend}")