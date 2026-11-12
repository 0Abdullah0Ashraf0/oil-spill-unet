"""U-Net implemented from scratch (no pretrained weights, no segmentation library)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.pool, self.conv = nn.MaxPool2d(2), DoubleConv(cin, cout)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin // 2, 2, stride=2)
        self.conv = DoubleConv(cin, cout)  # cin//2 (upsampled) + cin//2 (skip) = cin

    def forward(self, x, skip):
        x = self.up(x)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])  # odd input sizes
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """encoder -> bottleneck -> decoder with skip connections -> 1x1 conv to n_classes logits."""

    def __init__(self, in_ch=3, n_classes=5, base=32, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]  # e.g. 32,64,128,256,512
        self.inc = DoubleConv(in_ch, chs[0])
        self.downs = nn.ModuleList([Down(chs[i], chs[i + 1]) for i in range(depth)])  # last = bottleneck
        self.ups = nn.ModuleList([Up(chs[i + 1], chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], n_classes, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(skips[-1]))
        x = skips.pop()  # bottleneck output
        for up in self.ups:
            x = up(x, skips.pop())
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
