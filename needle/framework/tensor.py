from typing import overload, Optional
import numpy as np
import ctypes
import ml_dtypes


class Tensor:
    def __init__(self, data, dtype="float32", backend="cpu", **kwargs):
        self.backend = backend
        self.dtype = dtype
        self.ptr = None
        if backend == "cpu":
            self.data = np.array(data, dtype=dtype)
            self.shape = self.data.shape
            self.nelements = self.data.size
            self.nbytes = self.data.nbytes
            self.ptr = None  # Placeholder for device pointer if needed
        elif backend == "cuda":
            if isinstance(data, np.ndarray): # init a tensor from scratch
                self.data = data.astype(dtype)
                self.shape = data.shape
                self.nelements = data.size
                self.nbytes = data.nbytes
                from needle import ops
                self.ptr = ops.copy_tensor_h2d(self.data.ctypes.data_as(ctypes.c_void_p), self.nelements, np.dtype(dtype).itemsize)
            elif isinstance(data, int): # init a tensor from existing device pointer
                self.data = None
                self.shape = kwargs.get("shape", None)
                self.nelements = np.prod(self.shape) # type: ignore
                self.nbytes = self.nelements * np.dtype(dtype).itemsize
                self.ptr = data
            else:
                raise ValueError(f"tensor must be constructed from a np.ndarray or int (device pointer) for cuda backend, {type(data)} is not allowed")
    
    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, backend={self.backend}, value={self.data})"
    
    def numpy(self) -> Optional[np.ndarray]:
        if self.backend == "cpu":
            return self.data
        else:
            from needle import ops
            ptr = ops.copy_tensor_d2h(self.ptr, self.nelements, np.dtype(self.dtype).itemsize)
            array = np.frombuffer(ptr, dtype=self.dtype, count=self.nelements).reshape(self.shape)
            return array

    def cpu(self):
        if self.backend == "cpu":
            return self
        elif self.backend == "cuda":
            from needle import ops
            ptr = ops.copy_tensor_d2h(self.ptr, self.nelements, np.dtype(self.dtype).itemsize)
            array = np.frombuffer(ptr, dtype=self.dtype, count=self.nelements).reshape(self.shape)
            return Tensor(array, dtype=self.dtype, backend="cpu")
        else:
            raise NotImplementedError(f"Conversion to cpu for backend '{self.backend}' is not implemented.")

    def cuda(self):
        if self.backend == "cuda":
            return self
        elif self.backend == "cpu":
            array = np.array(self.data, dtype=self.dtype)
            return Tensor(array, dtype=self.dtype, backend="cuda")

    def __del__(self):
        if self.backend == "cuda":
            if self.ptr is not None:
                from needle import ops
                ops.free_tensor(self.ptr)
            if self.data is not None:
                del self.data
        else:
            if self.data is not None:
                del self.data

    def __call_backend_method(self, other, method_name):
        if self.backend != other.backend:
            raise ValueError("Cannot add tensors with different backends")
        cls_method_name = f"{method_name}_{self.backend}"
        method = getattr(Tensor, cls_method_name, None)
        if method is None:
            raise NotImplementedError(f"{method_name} for backend '{self.backend}' is not implemented.")
        return method(self, other)

    def __add__(self, other):
        return self.__call_backend_method(other, "add")
    
    def __sub__(self, other):
        return self.__call_backend_method(other, "sub")
    
    def __mul__(self, other):
        return self.__call_backend_method(other, "mul")
    
    def __matmul__(self, other):
        return self.__call_backend_method(other, "matmul")
    
    @classmethod
    def add_cpu(cls, x, y):
        return Tensor(x.data + y.data, backend="cpu")
    
    @classmethod
    def sub_cpu(cls, x, y):
        return Tensor(x.data - y.data, backend="cpu")
    
    @classmethod
    def mul_cpu(cls, x, y):
        return Tensor(x.data * y.data, backend="cpu")
    
    @classmethod
    def matmul_cpu(cls, x, y):
        return Tensor(x.data @ y.data, backend="cpu")
    

def zeros(shape, backend="cpu"):
    if backend == "cpu":
        import numpy as np
        return Tensor(np.zeros(shape), backend="cpu")
    else:
        raise NotImplementedError(f"{backend} backend not implemented yet")
    

def ones(shape, backend="cpu"):
    if backend == "cpu":
        import numpy as np
        return Tensor(np.zeros(shape), backend="cpu")
    else:
        raise NotImplementedError(f"{backend} backend not implemented yet")
    

def expand_dims(tensor, axis):
    if tensor.backend == "cpu":
        return expand_dims_cpu(tensor, axis)
    else:
        raise NotImplementedError(f"{tensor.backend} backend not implemented yet")
    

def expand_dims_cpu(tensor, axis):
    import numpy as np
    new = np.expand_dims(tensor.data, axis=axis)
    return Tensor(new, backend=tensor.backend)
    

def repeat(tensor, repeats, axis=0):
    if tensor.backend == "cpu":
        return repeat_cpu(tensor, repeats, axis)
    else:
        raise NotImplementedError(f"{tensor.backend} backend not implemented yet")
    

def repeat_cpu(tensor, repeats, axis=0):
    import numpy as np
    array = tensor.data
    while axis >= array.ndim:
        array = np.expand_dims(array, axis=array.ndim)
    new = np.repeat(array, repeats, axis=axis)
    return Tensor(new, backend=tensor.backend)


def equal(tensor_a, tensor_b, type=int):
    if tensor_a.backend == "cpu":
        return equal_cpu(tensor_a, tensor_b, type)
    else:
        raise NotImplementedError(f"{tensor_a.backend} backend not implemented yet")
    

def equal_cpu(tensor_a, tensor_b, type):
    import numpy as np
    equal = np.equal(tensor_a.data, tensor_b.data)
    if type == int:
        return Tensor(equal.astype(np.int64), backend=tensor_a.backend)
    elif type == bool:
        return Tensor(equal, backend=tensor_a.backend)
    else:
        raise ValueError("type must be int or bool")
    

def arange(start, stop=None, step=1, backend="cpu"):
    if stop is None:
        stop = start
        start = 0
    if backend == "cpu":
        import numpy as np
        return Tensor(np.arange(start, stop, step), backend="cpu")
    else:
        raise NotImplementedError(f"{backend} backend not implemented yet")
    

def sum(tensor, axis=None, keepdims=False):
    if tensor.backend == "cpu":
        return sum_cpu(tensor, axis, keepdims)
    else:
        raise NotImplementedError(f"{tensor.backend} backend not implemented yet")
    
def sum_cpu(tensor, axis=None, keepdims=False):
    import numpy as np
    return Tensor(np.sum(tensor.data, axis=axis, keepdims=keepdims), backend=tensor.backend)