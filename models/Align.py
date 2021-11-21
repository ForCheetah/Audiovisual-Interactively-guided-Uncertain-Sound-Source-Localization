import torch
import torch.nn as nn

class Align(nn.Module):
    def __init__(self, mode='train'):
        super(Align, self).__init__()
        self.mode = mode
        self.audioConv1 = nn.Conv2d(512, 1024, 1)  # [X,512, 1, 1]   -> [X,1024,1,1]
        self.audioConv2 = nn.Conv2d(1024, 512, 1)
        self.imageConv1 = nn.Conv2d(512, 1024, 1)  # [X,512, 14, 14]   -> [X,1024,14,14]
        self.imageConv2 = nn.Conv2d(1024, 512, 1)

    def forward(self, audio1, audio2, object1, object2):  # [X,512,1,1] [X,512,1,1] [X,512,14,14]
        object1 = self.imageConv1(object1)
        objectToAlgin1 = self.imageConv2(object1)
        object2 = self.imageConv1(object2)
        objectToAlgin2 = self.imageConv2(object2)
        audio1 = self.audioConv1(audio1)
        audioToAlgin1 = self.audioConv2(audio1)
        audio2 = self.audioConv1(audio2)
        audioToAlgin2 = self.audioConv2(audio2)

        return audioToAlgin1, audioToAlgin2, objectToAlgin1, objectToAlgin2  # [X,512, 1, 1] [X,512, 1, 1] [X,512, 14, 14]

if __name__ == '__main__':
    x = torch.rand(10, 30, 15, 15)
    xx = x.flatten(1)
    print(xx.shape)


