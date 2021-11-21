import torch
import torch.nn as nn
import torch.nn.functional as functional
import cv2
import numpy as np

# 设计的是整个图像分支 处理图像特征
# input :
class ImageBranch(nn.Module):
    """

    output: y :
    """
    def __init__(self):
        super(ImageBranch, self).__init__()
        self.maxPool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxLinear = nn.Linear(512, 512)
        self.avgLinear = nn.Linear(512, 512)
        self.sigmoid = nn.Sigmoid()
        self.epsilon = 0.2
        self.tau = 0.03

        self.clsConv = nn.Conv2d(in_channels=512, out_channels=11, kernel_size=2, stride=1, padding=1)  # [X,512,14,14]->[X,11,15,15]




    def forward(self, image, audio, path):
        image_norm = functional.normalize(image, dim=1)  # [X,512,14,14]


        B = audio.shape[0]
        audio_max = self.maxPool(audio).view(B, -1)  # [X,512]
        audio_avg = self.avgPool(audio).view(B, -1)  # [X,512]
        audio_max = self.maxLinear(audio_max)  # [X,512]
        audio_avg = self.avgLinear(audio_avg)  # [X,512]
        audio_max = functional.normalize(audio_max, dim=1)
        audio_avg = functional.normalize(audio_avg, dim=1)
        audio_maxAndPool = audio_avg + audio_max  # [X,512]

        attention = torch.einsum('ncqa,nchw->nqa', [image_norm, audio_maxAndPool.unsqueeze(2).unsqueeze(3)]).unsqueeze(1)  # [X,1,14,14]
        pos = self.sigmoid((attention - self.epsilon)/self.tau)
        imageAttFeat = torch.mul(image, pos)  # [X,512,14,14]
        image_class = self.clsConv(imageAttFeat)

        # audioFC = self.audioPool(audio).view(B, -1)
        # imageFC = self.imagePool(image_object).view(B, -1)  # [X,512]
        # y = self.catLinear(torch.cat((imageFC, audioFC), dim=1)).unsqueeze(2).unsqueeze(3)

        # self.visualize(pos, path)

        return imageAttFeat, image_class  # [X,512,14,14]  [X,11,15,15]

    def visualize(self, heatmap, path):
        print(path)
        heatmap = heatmap.squeeze(1).detach().cpu().numpy()
        i = 0
        for key in path.keys():
            img = cv2.imread(path.get(key)[0])
            heat = cv2.resize(heatmap[i], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            # heat = self.normalize_img(heat)
            heat = np.uint8(255 * heat)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            superimposed_img = heat * 0.4 + img
            cv2.imwrite('/data/whs/show/ex/' + key + 'yuan.png', img)
            cv2.imwrite('/data/whs/show/ex/' + key + 'heat.png', heat)
            cv2.imwrite('/data/whs/show/ex/'+key+'.png', superimposed_img)
            i += 1





