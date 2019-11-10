import torch as t
import torchvision

print(t.empty(2, 3))
print(t.rand(2, 3))

#output
# tensor([[1.0561e-38, 1.0286e-38, 1.0653e-38],
#         [1.0469e-38, 9.5510e-39, 9.9184e-39]])
# tensor([[0.0249, 0.4710, 0.0556],
#         [0.6873, 0.9130, 0.1853]])

print(t.rand(12, 13).size())
print(t.rand(12, 13).size()[1])
print(t.rand(12, 13).size(0))

#four ways: add
x, y = t.rand(2, 3), t.rand(2, 3)
z1 = x + y
z2 = t.add(x, y)
z3 = t.Tensor(2, 3)
t.add(x, y, out=z3)
print(z1, z2, z3)

z4 = x.add(y)
print(x)
x.add_(y)      #加 "_" 输出相加后的值
print(x)


#形状变换 类似Numpy 的reshape操作
x = t.rand(3, 4)
print(x)
print(x[:, 2:4])       #取第3~4列， 下标0开始
print(x[0:2, :])       #取第1~2行
print(x[0:2, 1:3])     #取1~2行， 2~3列相交区域

#Numpy转换为Tensor  调用 from_numpy
x = t.rand(2, 3)
y = x.numpy()
print(type(x), type(y))
x.add_(1)
print(x, y)
z = t.from_numpy(y)
print(type(z))


print("\nuse GPU")
if t.cuda.is_available():
    device = t.device("cuda")
    y = t.ones_like(x, device=device)
    x = x.to(device)
    x2 = x.to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", t.double))
