import torch
import torch.nn as nn

a = torch.tensor([[[[1]],[[2]],[[3]]],[[[4]],[[1]],[[3]]]]).float()
b = torch.rand(2, 3, 2,2)
c = torch.mul(a,b)

print(a.shape)
print(a)
print(b)
print(c)
