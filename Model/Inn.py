import torch.nn as nn
import torch

from torchsummary import summary


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio, blocktype='MN'):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        if blocktype == 'MN':
            self.bottleneckBlock = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, bias=False),
                # nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.ReflectionPad3d(1),
                nn.Conv3d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
                # nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, bias=False),
                # nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        return self.bottleneckBlock(x)


class SingleInn(nn.Module):
    def __init__(self, inp):
        super(SingleInn, self).__init__()
        self.theta_phi = nn.Sequential(
            nn.Conv3d(inp//2, inp//2, 3, bias=False),
            nn.ReflectionPad3d(1)
         )
        self.theta_rho = nn.Sequential(
            nn.Conv3d(inp//2, inp//2, 3, bias=False),
            nn.ReflectionPad3d(1)
         )

    def forward(self, x):
        z1, z2 = x[:, :x.shape[1] // 2, :, :, :], x[:, x.shape[1] // 2:x.shape[1], :, :, :]
        return torch.cat((z1 * torch.exp(self.theta_phi(z2)) + self.theta_rho(z2), z2), dim=1)


class DetailNode(nn.Module):
    def __init__(self, inp=8, oup=8, blocktype='MN'):
        super(DetailNode, self).__init__()
        self.theta_phi = InvertedResidualBlock(inp=inp, oup=oup, expand_ratio=2, blocktype=blocktype)
        self.theta_rho = InvertedResidualBlock(inp=inp, oup=oup, expand_ratio=2, blocktype=blocktype)
        self.theta_eta = InvertedResidualBlock(inp=inp, oup=oup, expand_ratio=2, blocktype=blocktype)
        self.shffleconv = nn.Conv3d(inp * 2, inp * 2, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separatefeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2, :, :, :], x[:, x.shape[1] // 2:x.shape[1], :, :, :]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separatefeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3, inp_dim=8, oup_dim=8, blocktype="MN"):
        super(DetailFeatureExtraction, self).__init__()
        modules = [DetailNode(inp=inp_dim, oup=oup_dim, blocktype=blocktype) for _ in range(num_layers)]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        # seperate feature into two parts
        z1, z2 = x[:, :x.shape[1] // 2, :, :, :], x[:, x.shape[1] // 2:x.shape[1], :, :, :]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


