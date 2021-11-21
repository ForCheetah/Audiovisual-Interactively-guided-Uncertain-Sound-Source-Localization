import h5py  #导入工具包
import numpy as np
import os
import torch
import torch.nn as nn



# each item : vid  object  class
def SaveObject(output, a):
    maxpool = nn.AdaptiveMaxPool2d((1, 1))
    f = h5py.File('/data/whs/checkpoint/Code/object.h5', 'a')
    firstObject = maxpool(output['firstImage']).squeeze(3).squeeze(2)  # [X,512]
    secondObject = maxpool(output['secondImage']).squeeze(3).squeeze(2)  # [X,512]
    # a = output['vids']
    if 'firstObject' not in f.keys():
        items_1 = f.create_dataset('firstObject', [firstObject.shape[0], 512], maxshape=[None, 512])
        items_1[0:firstObject.shape[0]] = firstObject.detach().cpu().numpy()
        items_2 = f.create_dataset('secondObject', [secondObject.shape[0], 512], maxshape=[None, 512])
        items_2[0:secondObject.shape[0]] = secondObject.detach().cpu().numpy()
        dt = h5py.string_dtype(encoding='utf-8', length=None)
        index = f.create_dataset('index', [len(a)], maxshape=[None], dtype=dt)
        index[0:len(a)] = a
    else:
        current_1 = f['firstObject']
        current_1.resize((current_1.shape[0] + firstObject.shape[0], 512))
        current_1[current_1.shape[0] - firstObject.shape[0]:current_1.shape[0]] = firstObject.detach().cpu().numpy()
        current_2 = f['secondObject']
        current_2.resize((current_2.shape[0] + secondObject.shape[0], 512))
        current_2[current_2.shape[0] - secondObject.shape[0]:current_2.shape[0]] = secondObject.detach().cpu().numpy()
        cu_index = f['index']
        cu_index.resize([len(cu_index) + len(a)])
        cu_index[len(cu_index) - len(a): len(cu_index)] = a

    f.close()

def ReadH5():
    f = h5py.File('/data/whs/object/object.h5', 'r')  # 打开h5文件
    print(f.keys())  # 可以查看所有的主键
    data_1 = f['firstObject']
    data_2 = f['secondObject']
    t = torch.tensor(data_1)
    print(t)
    print(data_2)
    print(t.shape)
    index = f['index']
    print(index[:])
    f.close()

if __name__ == '__main__':
    first = torch.rand(3, 512)
    second = torch.rand(3, 512)
    c = ['a', 'b', 'c']
    output = {'firstObject': first, 'secondObject': second, 'vids': c}
    SaveObject(output)

    ReadH5()




