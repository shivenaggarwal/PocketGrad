# PocketGrad

> [!NOTE]
> This is an ongoing experiment.
> A tiny, manually-wired backprop playground that *might* one day grow into a proper autodiff engine.
> If I survive till then.

---

## What is this

PocketGrad is a minimalist NumPy-backend autograd engine implemented from scratch in Python.

It doesn’t know what PyTorch is. It doesn’t want to.

It just wants to compute gradients and mind its business.

It uses:
- Reverse-mode autodiff (backpropagation)
- Operator overloading
- Computational graph construction
- Manual `.backward()` to propagate gradients

All powered by Python classes, NumPy arrays, and questionable choices.

---

## Why

Because understanding automatic differentiation is cooler than just using it.

Because PyTorch has two million lines of code and I needed something with fewer zeroes.

Because I enjoy pain and recursion.

And because sometimes the best way to learn how a neural net trains is to build the thing that does the training.

---

## Goals

### Current Goals
- [x] Build `Tensor` class with autograd
- [x] Implement `sum()` operation with gradient propagation
- [ ] Add basic tensor operations: `+`, `-`, `*`, `/`, `**`
- [ ] Enable `.backward()` for composed operations
- [ ] Build simple MLP using only this engine

### Near-Future Goals
- [ ] Add graph visualization for debugging
- [ ] Make it less ugly (code-wise)
- [ ] Add more math ops: `exp`, `tanh`, `relu`, `log`
- [ ] Add more unit tests
- [ ] Modularize into something semi-usable
- [ ] Build a minimal training framework (optimizers, loss, etc.)
- [ ] Clone PyTorch in 500 lines or less. (lol.)

---

## Inspirations

- Micrograd by Andrej Karpathy
- Tinygrad by George Hotz
- Autograd by Dougal Maclaurin et al.
