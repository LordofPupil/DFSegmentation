import torch
import numpy as np
import torch
from einops import rearrange
from torch import nn
from monai.networks.blocks import PatchEmbed
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import BasicLayer
from monai.networks.blocks import Convolution
from collections.abc import Sequence
from torchsummary import summary
from monai.networks.nets import SwinUNETR
from Model.Inn import DetailFeatureExtraction
from monai.networks.nets.unet import UNet
from monai.networks.nets.swin_unetr import SwinUNETR


def proj_out(x, normalize=False):
    if normalize:
        x_shape = x.size()
        if len(x_shape) == 5:
            n, ch, d, h, w = x_shape
            x = rearrange(x, "n c d h w -> n d h w c")
            x = F.layer_norm(x, [ch])
            x = rearrange(x, "n d h w c -> n c d h w")
        elif len(x_shape) == 4:
            n, ch, h, w = x_shape
            x = rearrange(x, "n c h w -> n h w c")
            x = F.layer_norm(x, [ch])
            x = rearrange(x, "n h w c -> n c h w")
    return x


class Segmentation_Swin_Encoder(nn.Module):
    def __init__(self,
                 in_channel: int = 1,
                 spatial_dim: int = 3,
                 patch_size: Sequence[int] = (2, 2, 2),
                 embed_dim=16,
                 num_head: Sequence[int] = (2, 2, 2),
                 num_block=3,
                 depth: Sequence[int] = (1, 1, 1)
                 ):
        super(Segmentation_Swin_Encoder, self).__init__()

        assert len(patch_size) == spatial_dim, "patch size length must be equal to spatial_dim"
        assert len(num_head) == num_block, "num_head length must be equal to spatial_dim"
        assert len(depth) == num_block, "depth length must be equal to num_block"

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channel,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
            spatial_dims=spatial_dim
        )
        self.dropout = nn.Dropout(p=0.0)
        self.encoder = nn.Sequential(
            *[BasicLayer(dim=embed_dim,
                         depth=depth[i],
                         num_heads=num_head[i],
                         window_size=(7, 7, 7),
                         drop_path=[0.0, 0.0],
                         mlp_ratio=4.0,
                         qkv_bias=False,
                         drop=0.0,
                         attn_drop=0.0,
                         norm_layer=nn.LayerNorm,
                         downsample=None,
                         use_checkpoint=False, ) for i in range(num_block)])

    def forward(self, x):
        out1 = self.patch_embed(x)
        out2 = self.dropout(out1)
        out3 = proj_out(out2)
        out4 = self.encoder(out3)
        return out4


# net = Segmentation_Swin_Encoder(
#     in_channel=1,
#     spatial_dim=3,
#     patch_size=[2, 2, 2],
#     embed_dim=16,
#     num_head=[2, 4],
#     num_block=2,
#     depth=[1, 1]
# ).to("cuda")
# summary(net, (1, 224, 224, 96))


class CovolutionWithPooling(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 spatial_dims: int = 3,
                 strides: int = 1,
                 kernel_size: int = 1,
                 adn_ordering="NDA",
                 act=("prelu", {"init": 0.2}),
                 dropout=0.1,
                 norm="batch",
                 pooltype="avg"
                 ):
        super(CovolutionWithPooling, self).__init__()
        self.pooling = nn.AvgPool3d(kernel_size=2, stride=2) if pooltype == "avg" else nn.MaxPool3d(kernel_size=2,
                                                                                                    stride=2)
        self.conv = Convolution(spatial_dims=spatial_dims,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                strides=strides,
                                kernel_size=kernel_size,
                                adn_ordering=adn_ordering,
                                act=act,
                                dropout=dropout,
                                norm=norm)
        self.pad = nn.ConstantPad3d(1,0)
    def forward(self, x):
        x1 = self.conv(x)
        if x1.shape[4]%2 ==1:
            x1 = self.pad(x1)
        x2 = self.pooling(x1)
        return x2


class LW_Feature_Extractor(nn.Module):
    def __init__(self,
                 channels: Sequence[int] = (16, 24, 32, 64),
                 spatial_dims: int = 3,
                 strides: Sequence[int] = (1, 1, 1),
                 kernel_size: Sequence[int] = (3, 3, 3),
                 adn_ordering="NDA",
                 act=("prelu", {"init": 0.2}),
                 dropout=0.1,
                 norm="batch",
                 num_block=3
                 ):
        super(LW_Feature_Extractor, self).__init__()
        # assert len(channels) == (num_block+1), "channels length must be equal to num_block-1"
        self.pooling = nn.AvgPool3d(kernel_size=3, stride=2)
        self.encoder = nn.Sequential(
            *[CovolutionWithPooling(spatial_dims=spatial_dims,
                                    in_channels=channels[i],
                                    out_channels=channels[i + 1],
                                    strides=strides[i],
                                    kernel_size=kernel_size[i],
                                    adn_ordering=adn_ordering,
                                    act=act,
                                    dropout=dropout,
                                    norm=norm,
                                    pooltype="avg") for i in range(num_block)])

    def forward(self, x):
        # print(x.shape)
        x1 = self.pooling(x)
        # print(x1.shape)
        x2 = self.encoder(x1)
        return x2


# net = LW_Feature_Extractor().to("cuda")
# summary(net, (16, 112, 112, 48))


class HF_Feature_Extractor(nn.Module):
    def __init__(self, num_layers=3, inp_dim=8, oup_dim=8, blocktype="MN"):
        super(HF_Feature_Extractor, self).__init__()
        self.layer = DetailFeatureExtraction(num_layers=num_layers,
                                             inp_dim=inp_dim,
                                             oup_dim=oup_dim,
                                             blocktype=blocktype)

    def forward(self, x):
        x = self.layer(x)
        return x


# net = HF_Feature_Extractor().to("cuda")
# summary(net, (16, 112, 112, 48))

# net = SwinUNETR(img_size=(224, 224, 96), in_channels=2, out_channels=1, depths=(2, 4, 2, 2)).to("cuda")
# summary(net, input_size=(2, 224, 224, 96))
