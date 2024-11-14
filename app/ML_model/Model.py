import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
# import torch.sigmoid as sigmoid

#MobileNet으로 추정
class MobileNet(nn.Module):
    def __init__(self, pretrained = True):
        super(MobileNet, self).__init__() #torch.nn.Module의 서브클래스를 만들 때 부모 클래스이 __init__() 메서드를 호출해야함
        mobilenet = models.mobilenet_v3_large(pretrained=pretrained)
        features = list(mobilenet.features.children())
        self.enc = nn.Sequential(*features[:18])
        self.dec = nn.Conv2d(960, 1, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(7 * 7, 1)

    def forward(self, x):
        enc = self.enc(x) # 인코더를 거침
        dec = self.dec(enc) # 입력 = 960 채널
        out_map = torch.sigmoid(dec)
        out = self.linear(out_map.view(-1, 7 * 7))
        out = torch.sigmoid(out)
        out = torch.flatten(out)
        return out


class DeePixBiS(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        dense = models.densenet161(pretrained=pretrained)
        features = list(dense.features.children())
        self.enc = nn.Sequential(*features[:8])
        self.dec = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(14 * 14, 1)

    def forward(self, x):
        enc = self.enc(x)
        dec = self.dec(enc)
        out_map = F.sigmoid(dec)
        # print(out_map.shape)
        out = self.linear(out_map.view(-1, 14 * 14))
        out = F.sigmoid(out)
        out = torch.flatten(out)
        return out_map, out
