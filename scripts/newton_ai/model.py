"""Small U-shaped residual estimator; predicts log-RGB interference, never image content."""

import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.layers(x)


class FringeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0 = Block(3, 12)
        self.e1 = Block(12, 24)
        self.e2 = Block(24, 40)
        self.e3 = Block(40, 64)
        self.d2 = Block(104, 40)
        self.d1 = Block(64, 24)
        self.d0 = Block(36, 12)
        self.head = nn.Conv2d(12, 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        a = self.e0(x)
        b = self.e1(F.avg_pool2d(a, 2))
        c = self.e2(F.avg_pool2d(b, 2))
        d = self.e3(F.avg_pool2d(c, 2))
        c = self.d2(
            torch.cat(
                (
                    c,
                    F.interpolate(
                        d, scale_factor=2, mode="bilinear", align_corners=False
                    ),
                ),
                1,
            )
        )
        b = self.d1(
            torch.cat(
                (
                    b,
                    F.interpolate(
                        c, scale_factor=2, mode="bilinear", align_corners=False
                    ),
                ),
                1,
            )
        )
        a = self.d0(
            torch.cat(
                (
                    a,
                    F.interpolate(
                        b, scale_factor=2, mode="bilinear", align_corners=False
                    ),
                ),
                1,
            )
        )
        values = self.head(a)
        # Average first: source grain and the finest texture are not correction targets.
        residual = F.interpolate(
            F.avg_pool2d(values[:, :3], 2),
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        return torch.cat((0.3 * torch.tanh(residual), torch.sigmoid(values[:, 3:4])), 1)
