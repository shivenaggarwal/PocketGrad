import unittest

from engine.tensor import Tensor

class TestTensorAdd(unittest.TestCase):
    def test_add(self):
        t1 = Tensor([1,2,3], requires_grad=True)
        t2 = Tensor([4,5,6], requires_grad=True)

        t3 = t1 + t2

        assert t3.data.tolist() == [5, 7, 9]

        t3.backward(Tensor([-1,-2,-3]))

        assert t1.grad.data.tolist() == [-1,-2,-3]
        assert t2.grad.data.tolist() == [-1,-2,-3]

        t1 += 0.1
        assert t1.grad is None
        assert t1.data.tolist() == [1.1, 2.1, 3.1]

    def test_broadcast_add(self):
        # if we do t1+t2 and t1.shape == t2.shape, add is simple
        # but we should be allowed to manipulate the shapes and add different shape tensors as well
        # for eg: t1.shape == (10,5) t2.shape==(5,) => then for t1+t2, t2 will be viewed as (1,5)
        # i.e. if t2 = [1,2,3] => t2 = [[1,2,3]]
        # other thing is if one tensor has a 1 in soem dim, then we can just expand that

        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)  # (2, 3)
        t2 = Tensor([7, 8, 9], requires_grad = True)               # (3,)

        t3 = t1 + t2   # shape (2, 3)

        assert t3.data.tolist() == [[8, 10, 12], [11, 13, 15]]

        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [2, 2, 2]

    def test_broadcast_add2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)    # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad = True)               # (1, 3)

        t3 = t1 + t2

        assert t3.data.tolist() == [[8, 10, 12], [11, 13, 15]]

        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [[2, 2, 2]]
