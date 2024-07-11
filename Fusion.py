import torch
from einops import rearrange
from torch import nn
from torchsummary import summary
import numbers
from Model.Inn import DetailFeatureExtraction, SingleInn


def to_3d(x):
    return rearrange(x, 'b c h w d-> b (h w d) c')


def to_4d(x, h, w, d):
    return rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w, d = x.shape[-3:]
        return to_4d(self.body(to_3d(x)), h, w, d)


class HF_Fusion(nn.Module):
    def __init__(self, num_layers=3, inp_dim=16, oup_dim=16, blocktype="MN"):
        super(HF_Fusion, self).__init__()
        self.layer = DetailFeatureExtraction(num_layers=num_layers, inp_dim=inp_dim, oup_dim=oup_dim,
                                             blocktype=blocktype)

    def forward(self, x):
        x = self.layer(x)
        return x


class LF_Fusion(nn.Module):
    def __init__(self, in_channels=128, out_channels=64):
        super(LF_Fusion, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class FSAS(nn.Module):
    def __init__(self, dim, bias=False):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)
        self.to_hidden_si = SingleInn(dim * 3)

        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_si(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) (d patch3) -> b c h w d patch1 patch2 patch3',
                            patch1=self.patch_size, patch2=self.patch_size, patch3=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) (d patch3) -> b c h w d patch1 patch2 patch3',
                            patch1=self.patch_size, patch2=self.patch_size, patch3=self.patch_size)
        q_fft = torch.fft.rfftn(q_patch.float())
        k_fft = torch.fft.rfftn(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfftn(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w d patch1 patch2 patch3 -> b c (h patch1) (w patch2) (d patch3)',
                        patch1=self.patch_size, patch2=self.patch_size)
        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output
