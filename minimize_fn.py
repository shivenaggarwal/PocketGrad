# minimizing a function using our tiny autograd

from engine.tensor import Tensor, mul

x = Tensor([11, -19, 7, -1, 2, 13], requires_grad=True)

# minimizing the sum of squares
for i in range(100):
    sum_of_squares = mul(x, x).sum()  # is a 0-tensor
    sum_of_squares.backward()
    # x -= 0.1 * x.grad
    delta_x = mul(Tensor(0.1), x.grad)
    x = Tensor(x.data - delta_x.data, requires_grad=True)

    print(i, sum_of_squares)
