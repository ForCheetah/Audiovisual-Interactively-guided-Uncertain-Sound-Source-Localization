"""
    利用NMS的原理实现目标的分离
    首先输入的是热力图 heatmap
    将heatmap 的背景去掉 变成0
    然后找最大值，将最大值周围 c*c 的位置变成0  继续找最大值
    最后根据距离留下两个最大值就好了
    根据最大值的位置 将heatmap拆成两个 输出两个 attention图

"""


import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np


class NMS(nn.Module):
    def __init__(self, threshold=0.6, r=5):
        super(NMS, self).__init__()
        self.length = 14
        self.threshold = threshold
        self.r = r

    def indexToCo(self, index):
        return index//self.length, index%self.length

    def area(self, index):
        x, y = self.indexToCo(index)
        x1 = x-self.r if x-self.r>0 else 0
        y1 = y-self.r if y-self.r>0 else 0
        x2 = x+self.r if x+self.r<15 else 15
        y2 = y+self.r if y+self.r<15 else 15
        return x1, y1, x2, y2

    def remove(self, heat, indexOfMax):  # [X,196] [X]
        heat = heat.view(indexOfMax.shape[0], self.length, self.length)
        for i in range(indexOfMax.shape[0]):
            x1, y1, x2, y2 = self.area(indexOfMax[i])
            heat[i][x1:x2,y1:y2] = 0
        return heat.view(indexOfMax.shape[0], self.length*self.length)


    def dist(self, co1, co2):
        return pow(co2[0]-co1[0], 2) + pow(co2[1]-co1[1], 2)

    # [4,X]   ->   [X,2,2]
    # 选出距离最远的两个中心
    def pickTwo(self, indexList):
        # 先转置
        indexList = np.array(indexList.cpu()).T  # [X,4]
        output = []  # [X,2,2]
        for i in range(indexList.shape[0]):
            list = []  # [4,2]
            dists = []
            for j in range(indexList.shape[1]):
                x, y = self.indexToCo(indexList[i][j])
                list.append([x, y])
            dists.append(self.dist(list[0], list[1]))
            dists.append(self.dist(list[0], list[2]))
            dists.append(self.dist(list[0], list[3]))
            dists.append(self.dist(list[1], list[2]))
            dists.append(self.dist(list[1], list[3]))
            dists.append(self.dist(list[2], list[3]))
            index = np.argmax(dists)
            if index == 0:
                output.append([list[0], list[1]])
            if index == 1:
                output.append([list[0], list[2]])
            if index == 2:
                output.append([list[0], list[3]])
            if index == 3:
                output.append([list[1], list[2]])
            if index == 4:
                output.append([list[1], list[3]])
            if index == 5:
                output.append([list[2], list[3]])

        return output


    def separateHeat(self, heatmap):
        pass


    # heatmap:[X,1,14,14]   -> output:[X,n,14,14]
    # 直接输出 两个 [X,14,14]不好么？
    def forward(self, heatmap):
        heatmap = heatmap.detach()
        heatmap = heatmap.squeeze(1)  # [X,14,14]
        indexList = None
        B, H, W = heatmap.shape
        mask = (heatmap > self.threshold).float()  # [X,14,14]
        heat = heatmap.mul(mask).view(B, H*W)  # [X,196]

        for i in range(4):
            indexOfMax = torch.argmax(heat, dim=1)  # [X]
            heat = self.remove(heat, indexOfMax)
            if indexList is None:
                indexList = indexOfMax.view(1, B)
            else:
                indexList = torch.cat((indexList, indexOfMax.view(1, B)), dim=0)
        centers = self.pickTwo(indexList)  # [X,2,2]

        # 下面直接生成两个二值的mask 叠上去就行了
        masks1 = None   # [X,14,14]
        masks2 = None
        for i in range(len(centers)):
            m1 = torch.zeros((H, W))
            center = centers[i]
            co1, co2 = center[0], center[1]
            for u in range(H):
                for v in range(W):
                    if self.dist([u, v], co1) < self.dist([u, v], co2):
                        m1[u][v] = 1
            m2 = (m1 < 0.9).float()
            m1 = m1.view(1, H, W)
            m2 = m2.view(1, H, W)
            if masks1 is None:
                masks1 = m1
                masks2 = m2
            else:
                masks1 = torch.cat((masks1, m1), dim=0)
                masks2 = torch.cat((masks2, m2), dim=0)

        masks1 = masks1.unsqueeze(1).cuda()
        masks2 = masks2.unsqueeze(1).cuda()

        return masks1, masks2  # [X,1,14,14]  [X,1,14,14]


if __name__ == "__main__":
    x = torch.rand((2, 1, 14, 14))
    nms = NMS()
    m1, m2 = nms(x)
    # 通过检验

