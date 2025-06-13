# minimizing a function using our tiny autograd

from engine.tensor import Tensor

x = Tensor([11, -19, 7, -1, 2, 13], requires_grad=True)

# minimizing the sum of squares
for i in range(100):
    x.zero_grad() # before inplace ops we were creating new tensors each time so we were zeroing the grads automatically but now we do it manually

    sum_of_squares = (x * x).sum()  # is a 0-tensor
    sum_of_squares.backward()
    # x -= 0.1 * x.grad
    delta_x = 0.1 * x.grad
    # x = Tensor(x.data - delta_x.data, requires_grad=True)
    x -= delta_x

    print(i, sum_of_squares)
