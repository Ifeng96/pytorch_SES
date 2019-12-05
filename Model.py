from torch import nn
from torchvision import models
import torch.nn.functional as F

class SElayer(nn.Module):
    def __init__(self, in_channel, reduction):
        super(SElayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sequeeze = nn.Sequential(
            nn.Linear(in_channel//reduction, in_channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel//reduction, in_channel//reduction, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.conv1(x)
        xsq = self.sequeeze(self.avgpool(x).view(b,c)).view(b,c,1,1)
        return x*xsq.expand_as(x)

# block of resnet18,34
class Basiclayer(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Basiclayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(in_place=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)

# block of resnet50+
class Bottlenecklayer(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottlenecklayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel//4, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel//4, out_channel//4, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel//4, out_channel, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)

class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_block):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channel, out_channel, 3, 2, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        layers = []
        for _ in range(n_block):
            layers.append(Basiclayer(out_channel, out_channel))
        self.basicblock = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.basicblock(out)
        return out


class Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        channels = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.conv1 = nn.Conv2d(3, 64, 7, 2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder1 = DecoderBlock(channels[3], channels[2], 3)
        self.decoder2 = DecoderBlock(channels[2], channels[1], 6)
        self.decoder3 = DecoderBlock(channels[1], channels[0], 4)
        self.decoder4 = DecoderBlock(channels[0], channels[0], 3)

        self.lastupsample = nn.ConvTranspose2d(channels[0], 32, 3, 12, 1, 1)
        self.lastbasicblock = Basiclayer(32, 32)
        self.out_to_class = nn.Conv2d(32, num_classes, 3, 1, 1)

    def forward(self, x):
        _, _, h, w = x.size()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)

        encoder1 = self.encoder1(out)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)

        decoder1 = self.decoder1(encoder4) + encoder3
        decoder2 = self.decoder2(decoder1) + encoder2
        decoder3 = self.decoder3(decoder2) + encoder1
        decoder4 = self.decoder4(decoder3)

        out = self.lastupsample(decoder4)
        out = self.lastbasicblock(out)
        out = self.out_to_class(out)
        return F.softmax(out, dim=1)