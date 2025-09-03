import needle
import numpy as np


def test_tensor_operations():
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    a = needle.tensor(data, dtype="float32", backend="cuda")
    b = a.cpu()
    assert b.numpy() == data
