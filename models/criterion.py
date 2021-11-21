import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight[n])
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight)

        return err

class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))

class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.pow(pred - target, 2))


class BCEWithLogitsLoss(BaseLoss):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy_with_logits(pred, target, weight=weight)

class CELoss(BaseLoss):
    def __init__(self):
        super(CELoss, self).__init__()

    def _forward(self, pred, target, weight=None):
        return F.cross_entropy(pred, target)


class L1_new(BaseLoss):
    def __init__(self):
        super(L1_new, self).__init__()

    def _forward(self, pred, target, weight=None):
        return F.l1_loss(pred, target)

# input visual:[X,512,1,1]  audio:[X,512,1,1]
class NCELoss(nn.Module):
    def __init__(self):
        super(NCELoss, self).__init__()
        self.t = 1e-10

    def forward(self, feature1, feature2, temperature=10):
        add = torch.exp(torch.cosine_similarity(feature1, feature2, dim=1) / temperature)
        pos = torch.sum(add)

        simi = torch.matmul(feature1, feature2.t())
        feature1_value = torch.norm(feature1, p=2, dim=1).unsqueeze(0).t()
        feature2_value = torch.norm(feature2, p=2, dim=1).unsqueeze(0)
        fenzi = torch.matmul(feature1_value, feature2_value) + self.t
        neg = torch.sum(torch.exp(simi / fenzi / temperature) ) + self.t

        out = -torch.log(pos / neg)
        return out


class InfoNCE(nn.Module):
    def __init__(self):
        super(InfoNCE, self).__init__()
        self.e = 0.001

    def forward(self, sim1, sim2, s_others):  # [X,1]  [X,1]  [X,1]   [X,X]
        s_object = torch.exp(sim1)
        loss = torch.log((self.e + s_object) / (s_object + torch.exp(sim2) + s_others))
        loss = -torch.mean(loss)
        return loss


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 1.0

    def forward(self, sim1, sim2, s_others):  # [X,1] [X,1]  [X,1]
        loss = torch.mean(sim2 + s_others - sim1 + self.margin)
        return loss

class LocalizeLoss(nn.Module):
    def __init__(self, opt):
        super(LocalizeLoss, self).__init__()
        self.opt = opt

    # imageAttFeat image_class
    def forward(self, imageToClass, label):  # label :[X]
        cls_logits = torch.mean(torch.mean(imageToClass, dim=2), dim=2)  # [X,15]
        loss = F.cross_entropy(cls_logits, label)
        return loss

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pre, target):
        target = target.squeeze(1)
        return self.bceloss(torch.sigmoid(pre), target)

class ObjectAudioAlignLoss_BACK(nn.Module):
    def __init__(self):
        super(ObjectAudioAlignLoss_BACK, self).__init__()

    def forward(self, object, audio):  # [X,512, 1, 1]  [X,512, 1, 1]
        B, C, _, _ = object.shape
        object = object.view(B, C)
        audio = audio.view(B , C)
        simi = F.cosine_similarity(object.unsqueeze(1), audio.unsqueeze(0), dim=2)
        simi = torch.exp(simi / 2)
        mask_pos = torch.eye(B, B).cuda()
        pos = simi * mask_pos
        loss = -torch.log((torch.sum(pos)+0.000001) / torch.sum(simi))

        return loss

class ObjectAudioAlignLoss(nn.Module):
    def __init__(self):
        super(ObjectAudioAlignLoss, self).__init__()
        self.margin = torch.tensor(1).cuda()

    def forward(self, object, audio):  # [X,512, 1, 1]  [X,512, 1, 1]
        B, C, _, _ = object.shape
        object = object.view(B, C)
        audio = audio.view(B , C)
        simi = F.cosine_similarity(object.unsqueeze(1), audio.unsqueeze(0), dim=2)
        simi = torch.exp(simi / 2)
        mask_pos = torch.eye(B, B).cuda()
        pos = torch.sum(simi * mask_pos) / B
        neg = torch.sum(simi * (1-mask_pos)) / (B * (B-1))
        loss = neg - pos

        return loss



class CEOfAudio(nn.Module):
    def __init__(self):
        super(CEOfAudio, self).__init__()

    # imageAttFeat image_class
    def forward(self, audioPre, label):  # label :[X]
        loss = F.cross_entropy(audioPre, label)  # [X,num_class]
        return loss

class CEOfImage(nn.Module):
    def __init__(self):
        super(CEOfImage, self).__init__()

    # imageAttFeat image_class
    def forward(self, imagePre, label):  # [X,15] label :[X]
        cls_logits = imagePre  # [X,15]
        loss = F.cross_entropy(cls_logits, label)
        return loss

class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()
        self.tau = 0.00001

    def forward(self, simi1, simi2, mask1, mask2):  # [X,14,14]  [X,1,14,14]
        mask1 = mask1.squeeze(1)
        mask2 = mask2.squeeze(1)
        pos1 = torch.sum(torch.mul(simi1, mask1)) / (torch.sum(mask1) + self.tau)
        pos2 = torch.sum(torch.mul(simi2, mask2)) / (torch.sum(mask2) + self.tau)
        mask_background = 1-mask1-mask2
        neg1 = torch.sum(torch.mul(simi1, 1-mask1)) / (torch.sum(1-mask1) + self.tau)
        neg2 = torch.sum(torch.mul(simi2, 1-mask2)) / (torch.sum(1-mask2) + self.tau)
        # neg3 = torch.mean(simiN)
        # loss = -torch.log((pos1 + pos2) / (neg1 + neg2))
        loss = 3.5 + neg1 + neg2 - pos1 - pos2
        return loss


# class SimilarityLoss(nn.Module):
#     def __init__(self):
#         super(SimilarityLoss, self).__init__()
#         self.epsilon = 0.75
#         self.tau = 0.03
#         self.e = 0.0001
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, simi1, simi2, neg1, neg2):  # [X,14,14]
#         simi1 = self.sigmoid(simi1)
#         simi2 = self.sigmoid(simi2)
#         neg1 = self.sigmoid(neg1)
#         neg2 = self.sigmoid(neg2)
#         r1 = self.sigmoid((simi1 - self.epsilon)/self.tau)
#         r2 = self.sigmoid((simi2 - self.epsilon)/self.tau)
#         neg1 = torch.mean(torch.mean(neg1, dim=2), dim=1)
#         neg2 = torch.mean(torch.mean(neg2, dim=2), dim=1)
#         rn1 = 1 - r1
#         rn2 = 1 - r2
#
#         pos1 = (r1 * simi1).view(*simi1.shape[:1], -1).sum(-1) / (r1.view(*r1.shape[:1], -1).sum(-1))  #[X]
#         pos2 = (r2 * simi2).view(*simi2.shape[:1], -1).sum(-1) / (r1.view(*r2.shape[:1], -1).sum(-1))
#         rneg1 = (rn1 * simi1).view(*simi1.shape[:1], -1).sum(-1) / (r1.view(*rn1.shape[:1], -1).sum(-1))  #[X]
#         rneg2 = (rn2 * simi1).view(*simi1.shape[:1], -1).sum(-1) / (r1.view(*rn2.shape[:1], -1).sum(-1))  # [X]
#         neg = torch.exp(neg1 + neg2 + rneg1 + rneg2) #[X]
#         pos = torch.exp(pos1 + pos2)
#         loss = -torch.mean(torch.log((pos + self.e) / (pos+neg)))
#
#         return loss



if __name__ == "__main__":
    model = InfoNCE()
    print("Model loaded.")
    sim1 = Variable(torch.rand(8,1))
    sim2 = Variable(torch.rand(8,1))
    sim = Variable(torch.rand(8,8))
    print("Image loaded.")

    # Run a feedforward and check shape
    c = model(sim1, sim2, sim)
    print(c.shape)

