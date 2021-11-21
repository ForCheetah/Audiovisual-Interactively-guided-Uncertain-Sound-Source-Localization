import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as functional
import math




# x = torch.rand(3,1,1,512)
# y = torch.rand(3,14,14,512)
#
# aa = functional.cosine_similarity(y , x, dim=3)
# print(aa[0])
# a = functional.sigmoid(aa)
# b = torch.sigmoid(aa)
# print(a[0])
# print(b[0])


# n = x.shape[0]
#
# out = functional.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2)
# mask_pos = torch.eye(n, n)
# print(out)
# simi = torch.exp(out / 2)
#
# pos = simi * mask_pos
#
#
#
# print(simi)
#
# print(pos)
# loss = -torch.log(torch.sum(pos) / torch.sum(simi))
# print(loss)
#
#
# a = torch.tensor(1)
# print(a)


