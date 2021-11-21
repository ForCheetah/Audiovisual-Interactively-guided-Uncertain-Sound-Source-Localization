import torch
import torch.nn as nn

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    return nn.Sequential(*[upconv, upnorm, uprelu])

class AudioSeparation(nn.Module):
    def __init__(self):
        super(AudioSeparation, self).__init__()
        self.audionet_convlayer1 = unet_conv(512, 512)
        self.audionet_convlayer2 = unet_conv(512, 512)
        self.maxPool = nn.AdaptiveMaxPool2d((1, 1))

        self.audionet_upconvlayer1 = unet_upconv(1024, 512)
        self.audionet_upconvlayer2 = unet_upconv(1024, 512)

        self.maxPool2 = nn.AdaptiveMaxPool2d((1, 1))
        self.maxPool3 = nn.AdaptiveMaxPool2d((1, 1))

        self.visualConv = nn.Conv2d(512, 512, 1)

    def forward(self, audio, visual):  # [X,512,16,16] [X,512,16,16]  ->  [X,512,16,16]
        down_1 = self.audionet_convlayer1(audio)
        down_2 = self.audionet_convlayer2(down_1)

        visualSignal = self.visualConv(visual)
        visualSignal = self.maxPool(visualSignal)  # [X,512,1,1]
        visualSignal = torch.mul(down_2, visualSignal)
        # visualSignal = visualSignal.repeat(1, 1, down_2.shape[2], down_2.shape[3])

        up_1 = self.audionet_upconvlayer1(torch.cat((down_2, visualSignal), dim=1))
        up_2 = self.audionet_upconvlayer2(torch.cat((up_1, down_1), dim=1))

        B, C, _, _ = up_2.shape
        audioToAlign = self.maxPool2(up_2)  # [X,512,1,1]

        return audioToAlign


if __name__ == '__main__':
    x = torch.rand(23,512,16,16)
    y = torch.rand(23,512,16,16)
    model = AudioSeparation()
    model(x, y)



