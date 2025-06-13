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
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    # clean string representation similar to that of numpy or pytorch
    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0) # if no grad is supplied and the tensor is a scalar, it defaults to 1
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

def add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    # idea: [1,2,3] + [4+e,5,6] = [5+e,7,9]
    # this basically means we get the same grad back in addition
    # but this doesnt handle broadcasting properly

    # handling broadcasting
    def reduce_grad(grad: np.ndarray, shape: tuple) -> np.ndarray:
        ndims_added = grad.ndim - len(shape)
        for _ in range(ndims_added): # summing added dims
            grad = grad.sum(axis=0)
        # summing across broadcastd (but not added dims)
        # eg: (2,3) + (1,3) => (2,3) grad(2,3)
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return reduce_grad(grad, t1.shape)
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return reduce_grad(grad, t2.shape)
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def mul(t1: Tensor, t2: Tensor) -> Tensor:
  # y = a*b
  # we know: dL/dy
  # dL/da = dL/dy * dy/da = dL/dy * b

  data = t1.data * t2.data
  requires_grad = t1.requires_grad or t2.requires_grad
  depends_on: List[Dependency] = []

  if t1.requires_grad:
    def grad_fn1(grad: np.ndarray) -> np.ndarray:
      grad = grad * t2.data

      # Sum out added dims
      ndims_added = grad.ndim - t1.data.ndim
      for _ in range(ndims_added):
        grad = grad.sum(axis=0)

      # Sum across broadcasted (but non-added dims)
      for i, dim in enumerate(t1.shape):
        if dim == 1:
          grad = grad.sum(axis=i, keepdims=True)

      return grad

    depends_on.append(Dependency(t1, grad_fn1))

  if t2.requires_grad:
    def grad_fn2(grad: np.ndarray) -> np.ndarray:
      grad = grad * t1.data

      # Sum out added dims
      ndims_added = grad.ndim - t2.data.ndim
      for _ in range(ndims_added):
        grad = grad.sum(axis=0)

      # Sum across broadcasted (but non-added dims)
      for i, dim in enumerate(t2.shape):
        if dim == 1:
          grad = grad.sum(axis=i, keepdims=True)

      return grad

    depends_on.append(Dependency(t2, grad_fn2))

  return Tensor(data, requires_grad, depends_on)

def neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def sub(t1: Tensor, t2: Tensor) -> Tensor:
    return add(t1, neg(t2))
