import numpy as np
from typing import List, NamedTuple, Callable, Optional, Union

# each tensor can depend on other tensors. this dependency object records which tensor it depends on
# and it also has a grad_fn which describes how the gradient should be backpropped
class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[float, list, np.ndarray]

# this is just an helper function that casts floats, lists, etc to numpy array for better internal representation
def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)

class Tensor:
    def __init__(self, data: Arrayable, requires_grad: bool = False, depends_on: List[Dependency] = None) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    # clears the gradient to zeros. useful before reusing the tensor in a new backward pass
    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    # clean string representation similar to that of numpy or pytorch
    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1) # if no grad is supplied and the tensor is a scalar, it defaults to 1
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data += grad.data # accumulates the gradient (important for branches in computational graph)

        # recursively calls backward() on all dependencies, using their grad_fn to compute appropriate local gradients
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    # a wrapper function that delegates to a function style of tensor_sum
    def sum(self) -> 'Tensor':
        return tensor_sum(self)

# this function here takes a tensor and returns the 0-tensor that's the sum of all its elements
# basically performs a full reduction sum of the tensor’s data (returns a scalar)
def tensor_sum(t: Tensor) -> Tensor:
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        # creates a grad_fn that maps the incoming scalar gradient back to the original shape
        # why this works: d(sum)/d(element) = 1, so gradient is broadcasted
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []

    # constructs the new scalar tensor, carrying the necessary backward metadata
    return Tensor(data, requires_grad, depends_on)
