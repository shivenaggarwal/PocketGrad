import unittest
from engine.tensor import Tensor

class TestTensorSum(unittest.TestCase):
    def test_simple_sum(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1.sum() # computes the sum of [1,2,3] -> 6

        t2.backward() # backpropagates from scalar output

        assert t1.grad.data.tolist() == [1, 1, 1] # checks that t1.grad equals [1, 1, 1] (as expected: d(sum)/d(xi) = 1)

    def test_sum_with_grad(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1.sum()

        t2.backward(Tensor(3)) # manually passes in gradient 3 instead of defaulting to 1

        assert t1.grad.data.tolist() == [3, 3, 3] # checks that the final gradients become [3, 3, 3] due to scalar multiplication of upstream gradient
