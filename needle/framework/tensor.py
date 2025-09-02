class Tensor:
    def __init__(self, data, backend="cpu"):
        if backend == "cpu":
            import numpy as np
            self.data = np.array(data)
            self.shape = self.data.shape
            self.backend = backend
            self.size = self.data.size
            self.nbytes = self.data.nbytes
            self.ptr = None  # Placeholder for device pointer if needed
        elif backend == "cuda":
            raise NotImplementedError("CUDA backend not implemented yet")

    def to_cuda(self):
        pass

    def __add__(self, other):
        if self.backend != other.backend:
            raise ValueError("Cannot add tensors with different backends")
        func = getattr(self, f"__add__{self.backend}", None)
        if func is None:
            raise NotImplementedError(f"Addition for backend '{self.backend}' is not implemented.")
        return func(other)
    
    def __sub__(self, other):
        if self.backend != other.backend:
            raise ValueError("Cannot subtract tensors with different backends")
        return Tensor(self.data - other.data, backend=self.backend)
    
    def __mul__(self, other):
        if self.backend != other.backend:
            raise ValueError("Cannot multiply tensors with different backends")
        return Tensor(self.data * other.data, backend=self.backend)
    
    def __matmul__(self, other):
        if self.backend != other.backend:
            raise ValueError("Cannot matmul tensors with different backends")
        return Tensor(self.data @ other.data, backend=self.backend)
    
    @staticmethod
    def __add__cpu(x, y):
        return Tensor(x.data + y.data, backend="cpu")