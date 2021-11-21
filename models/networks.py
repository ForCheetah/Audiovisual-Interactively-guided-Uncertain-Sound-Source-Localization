import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])


def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))

    if(Relu):
        model.append(nn.ReLU())

    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

# 修改了最后一层的 resnet18
class Resnet18(nn.Module):
    def __init__(self, original_resnet, pool_type='maxpool', input_channel=3, with_fc=False, fc_in=512, fc_out=512):
        super(Resnet18, self).__init__()
        self.pool_type = pool_type
        self.input_channel = input_channel
        self.with_fc = with_fc

        #customize first convolution layer to handle different number of channels for images and spectrograms
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = [self.conv1]
        layers.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers) #features before pooling
        #print self.feature_extraction

        if pool_type == 'conv1x1':
            self.conv1x1 = create_conv(512, 128, 1, 0)
            self.conv1x1.apply(weights_init)

        if with_fc:
            self.fc = nn.Linear(fc_in, fc_out)
            self.fc.apply(weights_init)

    def forward(self, x):
        x = self.feature_extraction(x)

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)
        elif self.pool_type == 'conv1x1':
            x = self.conv1x1(x)
        else:
            return x #no pooling and conv1x1, directly return the feature map

        if self.with_fc:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            if self.pool_type == 'conv1x1':
                x = x.view(x.size(0), -1, 1, 1) #expand dimension if using conv1x1 + fc to reduce dimension
            return x
        else:
            return x

# 设计了两个U-net结构可供选用
class UnetDown(nn.Module):
    def __init__(self, ngf=64, input_nc=1):
        super(UnetDown, self).__init__()


        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = unet_conv(ngf * 8, ngf * 8)



    def forward(self, x):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        audioFeature = [audio_conv1feature, audio_conv2feature, audio_conv3feature,
                        audio_conv4feature, audio_conv5feature, audio_conv6feature,
                        audio_conv7feature]

        return audioFeature



class UnetUp(nn.Module):
    def __init__(self, ngf=64, output_nc=1):
        """
        create Unet upConv
        :param ngf:
        :param output_nc:
        """
        super(UnetUp, self).__init__()

        self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

    def forward(self, audioFeature, imageAttFeat):  # imageAttFeat [X,512,1,1]
        imageAttFeat = imageAttFeat.repeat(1, 1, audioFeature[6].shape[2], audioFeature[6].shape[3])  # [X,512,2,2]
        imageAudioFeat = torch.cat((imageAttFeat, audioFeature[6]), dim=1)

        audio_upconv1feature = self.audionet_upconvlayer1(imageAudioFeat)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audioFeature[5]), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audioFeature[4]), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audioFeature[3]), dim=1))
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audioFeature[2]), dim=1))
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audioFeature[1]), dim=1))
        mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audioFeature[0]), dim=1))
        return mask_prediction


class ImageFusion(nn.Module):
    def __init__(self):
        super(ImageFusion, self).__init__()
        self.conv = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=2, padding=0, stride=2)
        self.maxPool = nn.AdaptiveMaxPool2d((1, 1))
        self.imageLinear = nn.Linear(1024, 768)

        self.audioPool = nn.AdaptiveMaxPool2d((1, 1))
        self.audioLinear = nn.Linear(512, 512)
        self.fusionLinear = nn.Linear(1280, 512)


    def forward(self, image, audio):
        """
        fusion the attention image and audio
        :param image: [X,512,14,14]
        :param audio: [X,512,2,2]
        :return: [X,512,1,1]
        """
        image = self.conv(image)  # [X,1024,6,6]
        B = image.shape[0]
        imageFC = self.maxPool(image).view(B, -1)  # [X,1024]
        imageFC = self.imageLinear(imageFC)  # [X,768]

        audioFC = self.audioPool(audio).view(B, -1)
        audioFC = self.audioLinear(audioFC)
        fusion = self.fusionLinear(torch.cat((imageFC, audioFC), dim=1)).unsqueeze(2).unsqueeze(3)  # [X,512,1,1]

        return fusion












