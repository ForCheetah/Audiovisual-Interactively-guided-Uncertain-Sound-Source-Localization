import torch
import torch.nn as nn



class Classify(nn.Module):
    def __init__(self):
        super(Classify, self).__init__()
        self.clsConv = nn.Conv2d(in_channels=512, out_channels=21, kernel_size=2, stride=1,
                                 padding=1)  # [X,512,14,14]->[X,15,15,15]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, image):
        imageToClassify = self.clsConv(image)
        imageToClassify = self.avgpool(imageToClassify)
        return imageToClassify.flatten(1)






def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# [X,512,16,16]  -> [X,512,16,16] -> [X,num_class]
class ClassifyOnAudio_Back(nn.Module):
    def __init__(self, block, num_class=30):
        super(ClassifyOnAudio_Back, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.groups = 1
        self.num_class = num_class
        self.base_width = 64
        self.layer5 = self._make_layer(block, 512, 2, stride=1)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        layers = []
        layers.append(block(512, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        for _ in range(1, blocks):
            layers.append(block(512, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x2 = self.layer5(x)
        B, C, _, _ = x2.shape
        x3 = self.avgPool(x2).view(B, C)
        out = self.fc(x3)
        return out


# [X,512]  ->  [X,num_class]
class ClassifyOnAudio(nn.Module):
    def __init__(self, num_class=30):
        super(ClassifyOnAudio, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_class)


    def forward(self, x):
        x = x.squeeze(3).squeeze(2)
        x = self.fc1(x)
        out = self.fc2(x)
        return out


class ClassifyOnImage(nn.Module):  # [X,512,14,14] -> [X,30]
    def __init__(self, num_class=15):
        super(ClassifyOnImage, self).__init__()
        # [X,512,14,14]->[X,512,1,1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # [X,512,1,1]->[X,30,1,1]
        self.clsConv = nn.Conv2d(in_channels=512, out_channels=num_class, kernel_size=1)

    def forward(self, image):
        image = self.avgpool(image)
        image = self.clsConv(image)
        return image.flatten(1)  # [X,30]

if __name__ == '__main__':
    x = torch.rand(10,30,15,15)
    xx = x.flatten(1)
    print(xx.shape)


