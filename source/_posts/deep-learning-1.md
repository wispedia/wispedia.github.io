---
title: 动手学深度学习（一）
date: 2023-08-03 20:09:49
mathjax: true
tags: [深度学习, Pytorch]
categories: 计算机
---

## 前言

本文章是来自于[动手学深度学习v2](https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497)的学习笔记

## 预备知识

***张量（tensor）***指的是n维数组，只有一个轴上的张量对应数学上的***向量（vector）***,具有两个轴的张量对应数学上的***矩阵（martix）***.

### 在Pytorch中有关张量的读写操作

```python
import torch
# 创建一个一维张量
x = torch.arange(12)

# 访问张量的形状
x.shape

# 张量中元素的总数
x.numel()

# 改变张量的形状, 使用-1可以自动计算出另一个维度，如 x.reshape(-1, 4)
x.reshape(3, 4)

# 创建值全为0的张量
torch.zeros((2, 3, 4))

# 创建值全为1的张量
torch.ones((2, 3, 4))

# 以下代码创建一个形状为（3,4）的张量。 其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样
torch.randn(3, 4)

# 创建一个确定值的张量
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

### 在Pytorch中有关张量的运算操作

```python
import torch

x = torch.sensor([1, 2, 4, 8])
y = torch.sensor([2, 2, 2, 2])

# 加 减 乘 除 求幂
x+y, x-y, x*y, x/y, x**y

# e的x幂运算
torch.exp(x)

# 所有元素求和
x.sum()

# 张量的连接
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# dim=0 按照形状第一个维度连接
torch.cat((X, Y), dim=0)
# (tensor([[ 0.,  1.,  2.,  3.],
#          [ 4.,  5.,  6.,  7.],
#          [ 8.,  9., 10., 11.],
#          [ 2.,  1.,  4.,  3.],
#          [ 1.,  2.,  3.,  4.],
#          [ 4.,  3.,  2.,  1.]]),

# dim=1 形状的第二个维度连接
torch.cat((X, Y), dim=1)

#  tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
#          [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
#          [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))

X == Y

# tensor([[False,  True, False,  True],
#         [False, False, False, False],
#         [False, False, False, False]])
```

### 广播机制

在某些情况下及时两个张量的形状不同，我们也可以通过调用***广播机制***来执行元素操作。这种机制为：

1. 通过适当的复制元素来扩展一个或者两个数组，使得两个张量具有相同的形状
2. 对新生成的数组执行按元素操作

```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))

# 由于a和b的形状不相同，所以在相加时会先通过广播机制将两个张量扩大到3x2的张量之后，
# 再进行运算, 扩展的行或者列使用默认值0来代替
a + b
# tensor([[0, 1],
#         [1, 2],
#         [2, 3]])
```

### 索引和切片

```python
X = torch.arange(12, dtype=torch.float32).reshape((3,4))

# 获取最后一个元素
X[-1]

# 指定索引更改张量中数据
X[1, 2] = 9

# 获取第一行和第二行的所有元素
X[0:2, :]
```