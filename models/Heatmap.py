import torch
import torch.nn as nn
import torch.nn.functional as functional


class Heatmap(nn.Module):
    """

    output: y :
    """
    def __init__(self):
        super(Heatmap, self).__init__()
        self.maxPool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxLinear = nn.Linear(512, 512)
        self.avgLinear = nn.Linear(512, 512)
        self.sigmoid = nn.Sigmoid()
        self.epsilon = 0.75
        self.tau = 0.03
        self.e = 0.0001

        self.clsConv = nn.Conv2d(in_channels=512, out_channels=11, kernel_size=2, stride=1, padding=1)

    def forward(self, image, audio, vids):
        B = image.shape[0]
        self.mask = (1 - 100 * torch.eye(B, B)).cuda()
        img = nn.functional.normalize(image, dim=1)
        img = functional.dropout(img, p=0.8, training= True)

        # Audio
        aud = self.maxPool(audio).view(B, -1)
        aud = nn.functional.normalize(aud, dim=1)

        # Join them
        A = torch.einsum('ncqa,nchw->nqa', [img, aud.unsqueeze(2).unsqueeze(3)]).unsqueeze(1)  # [X,1,14,14]
        A0 = torch.einsum('ncqa,ckhw->nkqa', [img, aud.t().unsqueeze(2).unsqueeze(3)])  # [X,X,14,14]

        # trimap
        Pos = self.sigmoid((A - self.epsilon) / self.tau)
        Pos2 = self.sigmoid((A - self.epsilon) / self.tau)
        Neg = 1 - Pos2

        Pos_all = self.sigmoid((A0 - self.epsilon) / self.tau)  # [14,14]

        # positive
        sim1 = (Pos * A).view(*A.shape[:2], -1).sum(-1) / (Pos.view(*Pos.shape[:2], -1).sum(-1))
        # negative
        sim = ((Pos_all * A0).view(*A0.shape[:2], -1).sum(-1) / Pos_all.view(*Pos_all.shape[:2], -1).sum(
            -1)) * self.mask
        sim2 = (Neg * A).view(*A.shape[:2], -1).sum(-1) / Neg.view(*Neg.shape[:2], -1).sum(-1)

        s_others = torch.exp(torch.mean(sim, dim=1)).view(sim.shape[0], 1)


        return A, sim1, sim2, s_others  # [X,1] [X,1] [X,1]




