from abc import ABC
from typing import Any

class Module(ABC):
    def __init__(self, prefix="", dtype="float32", backend="cpu"):
        self.prefix = prefix
        self.backend = backend
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        model forward function that dispatches to the correct backend-specific implementation.
        """
        return self.forward(*args, **kwargs)
    
    def load_weights(self, prefix: str, *args, **kwargs) -> None:
        """
        recursively load weights for this module and its submodules.
        """
        recursive_prefix = f"{prefix}.{self.prefix}" if self.prefix else prefix
        load_weights_method = getattr(self, f"weight_loader_{self.backend}", None)
        if load_weights_method is None:
            raise NotImplementedError(f"Weight loader for backend '{self.backend}' is not implemented.")
        load_weights_method(recursive_prefix, *args, **kwargs)
        for attr_key, attr_value in self.__dict__.items():
            if isinstance(attr_value, Module):
                attr_value.load_weights(recursive_prefix, *args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def weight_loader_cpu(self, prefix: str, *args, **kwargs):
        pass

    def weight_loader_cuda(self, prefix: str, *args, **kwargs):
        pass

    def weight_loader(self, prefix: str):
        pass