# PocketGrad

> [!NOTE]
> This is an ongoing experiment.
> A tiny, manually-wired backprop playground that *might* one day grow into a proper autodiff engine.
> If I survive till then.

---

## What is this

PocketGrad is a minimalist scalar-based autograd engine implemented from scratch in Python.

It doesn’t know what a tensor is. It doesn’t want to.

It just wants to compute gradients and mind its business.

It uses:
- Reverse-mode autodiff (backpropagation)
- Operator overloading
- Computational graph construction
- Manual `.backward()` to propagate gradients

All powered by Python classes, math, and questionable choices.

---

## Why

Because understanding automatic differentiation is cooler than just using it.

Because PyTorch has two million lines of code and I needed something with fewer zeroes.

Because I enjoy pain and recursion.

And because sometimes the best way to learn how a neural net trains is to build the thing that does the training.

---

## Goals

### Current Goals
- [ ] Implement scalar `Value` class with autograd
- [ ] Support basic operations: `+`, `-`, `*`, `/`, `**`
- [ ] Enable `.backward()` for reverse-mode autodiff
- [ ] Build a simple MLP using only scalar values
- [ ] Manually train it with a loss function and backprop

### Near-Future Goals
- [ ] Add graph visualization for debug/depression
- [ ] Make it less ugly (code-wise)
- [ ] Add more math ops: `exp`, `tanh`, `relu`, `log`
- [ ] Add unit tests to catch things before regret kicks in

### Long-Term Delusions
- [ ] Scale up to tensor-based engine with NumPy backend
- [ ] Modularize into something semi-usable
- [ ] Build a minimal training framework (optimizers, loss, etc.)
- [ ] Clone PyTorch in 500 lines or less. (lol.)

---


## Inspirations

- Micrograd by Andrej Karpathy
- Tinygrad by George Hotz
