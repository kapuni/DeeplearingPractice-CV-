import torch as t

a = t.rand(2, 3 , requires_grad=True)
b = t.rand(2, 3)
c = 2*a + b

print(a.grad_fn, b.grad_fn, c.grad_fn)

grandients = t.tensor([[0.1, 1.0, 0.001],
                      [0.1, 10, 1.0]], )
c.backward(grandients)
print(a.grad)
