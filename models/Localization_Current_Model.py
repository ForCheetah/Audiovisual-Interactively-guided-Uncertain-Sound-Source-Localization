import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import warpgrid
from models.NMS import NMS


class LocalizationModel(torch.nn.Module):
    def name(self):
        return 'LocalizationModel'

    def __init__(self, nets, opt):
        super(LocalizationModel, self).__init__()
        self.opt = opt
        self.mode = opt.mode
        self.num_cluster = opt.num_cluster
        self.maxPool = nn.AdaptiveMaxPool2d((1, 1))
        self.nms = NMS()
        self.net_image_extract, self.net_audio_extract, self.net_AVattention, \
        self.net_Audio_Separation, self.net_Align, self.net_ClassifyOnAudio, self.net_ClassifyOnImage = nets

    # input image: [X,3,224,224]  ->  [X,512,14,14]
    # input audio: [X,1,512,256]  ->  [X,512,16,16]

    def forward(self, input):

        vids = input['vid']
        audio = input['audio']
        image = input['image']
        pseudo_label1 = input['pseudo_label1']
        pseudo_label2 = input['pseudo_label2']



        # warp the spectrogram
        B = audio.size(0)
        T = audio.size(3)
        if self.opt.log_freq:  # whether use log-scale frequency
            grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).to(self.opt.device)
            audio = F.grid_sample(audio, grid_warp)

        audiofeature = self.net_audio_extract(audio)  # [X,512,16,16]
        imagefeature = self.net_image_extract(image)

        heatmap, sim1, sim2, s_others = self.net_AVattention(imagefeature, audiofeature, vids)  # heatmap:[X,1,14,14]

        imageAtten = torch.mul(imagefeature, heatmap)  # [X,512,14,14]    [X,512,14,14]*[X,1,14,14]

        mask_firstObject, mask_secondObject = self.nms(heatmap)  # [X,1,14,14]  [X,1,14,14]
        firstImage = torch.mul(imageAtten, mask_firstObject)  # [X,512,14,14]
        secondImage = torch.mul(imageAtten, mask_secondObject)  # [X,512,14,14]

        #
        firstAudio = self.net_Audio_Separation(audiofeature, firstImage)  # [X,512,1,1]  [X,512,1,1]
        secondAudio = self.net_Audio_Separation(audiofeature, secondImage)

        # [X,512, 1, 1] [X,512, 1, 1] [X,512, 14, 14]
        firstAudioToAlign, secondAudioToAlign, imageToAlign1, imageToAlign2 = self.net_Align(firstAudio, secondAudio,
                                                                                             firstImage, secondAudio)

        # 接下来要计算相似度 用余弦相似度
        B, C, _, _ = firstAudioToAlign.shape
        firstAudioToAlign = firstAudioToAlign.view(B, 1, 1, C)  # [X,1,1,512]
        secondAudioToAlign = secondAudioToAlign.view(B, 1, 1, C)  # [X,1,1,512]
        imageToAlign1 = imageToAlign1.permute(0, 2, 3, 1)  # [X,14,14,512]
        imageToAlign2 = imageToAlign2.permute(0, 2, 3, 1)  # [X,14,14,512]
        simi1 = F.cosine_similarity(imageToAlign1, firstAudioToAlign, dim=3)  # [X,14,14]
        simi2 = F.cosine_similarity(imageToAlign2, secondAudioToAlign, dim=3)  # [X,14,14]
        neg1 = F.cosine_similarity(imageToAlign1, firstAudioToAlign, dim=3)
        neg2 = F.cosine_similarity(imageToAlign2, secondAudioToAlign, dim=3)  # [X,14,14]
        mask1 = torch.mul(heatmap, mask_firstObject)  # [X,1,14,14]
        mask2 = torch.mul(heatmap, mask_secondObject)  # [X,1,14,14]

        imagePC1 = self.net_ClassifyOnImage(torch.mul(firstImage, simi1.unsqueeze(1)))  # [X,30]
        imagePC2 = self.net_ClassifyOnImage(torch.mul(secondImage, simi2.unsqueeze(1)))
        audioPC1 = self.net_ClassifyOnAudio(firstAudio)  # [X,30]
        audioPC2 = self.net_ClassifyOnAudio(secondAudio)

        Difference1 = torch.mean(torch.abs(imagePC1 - audioPC1), dim=1)  # [X]
        weight1 = torch.exp(-Difference1)
        Difference2 = torch.mean(torch.abs(imagePC2 - audioPC2), dim=1)
        weight2 = torch.exp(-Difference2)

        output = {'sim1': sim1, 'sim2': sim2, 's_others': s_others, 'firstImage': firstImage,
                  'secondImage': secondImage, 'vids': vids, 'heatmap': heatmap, 'imagePC1': imagePC1,
                  'imagePC2': imagePC2, 'audioPC1': audioPC1, 'audioPC2': audioPC2, 'pseudo_label1': pseudo_label1,
                  'pseudo_label2': pseudo_label2, 'simi1': simi1, 'simi2': simi2, 'neg1': neg1, 'neg2': neg2,
                  'weight1': weight1, 'weight2': weight2, 'mask1': mask1, 'mask2': mask2}

        # when test
        if self.mode == 'test':
            output['heatmap1'] = simi1
            output['heatmap2'] = simi2

        return output

