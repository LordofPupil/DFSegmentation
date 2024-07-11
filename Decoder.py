import torch
from monai.networks.nets.swin_unetr import BasicLayer
from torch import nn
from monai.networks.blocks import Convolution
import numpy as np
from collections.abc import Sequence
from torchsummary import summary


def get_padding(kernel_size: Sequence[int] | int, stride: Sequence[int] | int) -> tuple[int, ...] | int:
    """
    Get padding based on kernel size for TranspConvolution
    :param kernel_size:
    :param stride:
    :return:
    """
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
        kernel_size: Sequence[int] | int, stride: Sequence[int] | int, padding: Sequence[int] | int
) -> tuple[int, ...] | int:
    """
    Get output padding from input kernel size and stride for TranspConvolution
    :param kernel_size:
    :param stride:
    :param padding:
    :return:
    """
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


class Segmentation_Decoder(nn.Module):
    def __init__(self,
                 channels: Sequence[int] = (64, 32, 16),
                 spatial_dims=3,
                 exp=0
                 ):
        super(Segmentation_Decoder, self).__init__()

        self.exp = exp
        if exp == 0:
            self.transp_conv1 = Convolution(
                spatial_dims=spatial_dims,
                in_channels=channels[0],
                out_channels=channels[1],
                strides=4,
                kernel_size=4,
                norm="instance",
                dropout=0.0,
                bias=False,
                conv_only=True,
                is_transposed=True,
                padding=get_padding(4, 4),
                output_padding=get_output_padding(4, 4, get_padding(4, 4)),
            )
            self.transp_conv2 = Convolution(
                spatial_dims=spatial_dims,
                in_channels=channels[1],
                out_channels=channels[2],
                strides=4,
                kernel_size=4,
                norm="instance",
                dropout=0.0,
                bias=False,
                conv_only=True,
                is_transposed=True,
                padding=get_padding(4, 4),
                output_padding=get_output_padding(4, 4, get_padding(4, 4)),
            )
            self.swin = BasicLayer(dim=8,
                                   depth=2,
                                   num_heads=2,
                                   window_size=(7, 7, 7),
                                   drop_path=[0.0, 0.0],
                                   mlp_ratio=4.0,
                                   qkv_bias=False,
                                   drop=0.0,
                                   attn_drop=0.0,
                                   norm_layer=nn.LayerNorm,
                                   downsample=None,
                                   use_checkpoint=False, )
            self.transp_conv3 = Convolution(
                spatial_dims=spatial_dims,
                in_channels=8,
                out_channels=1,
                strides=2,
                kernel_size=2,
                norm="instance",
                dropout=0.0,
                bias=False,
                conv_only=True,
                is_transposed=True,
                padding=get_padding(2, 2),
                output_padding=get_output_padding(2, 2, get_padding(2, 2)),
            )
            self.conv1 = Convolution(in_channels=48, out_channels=24, kernel_size=1, spatial_dims=3)
            self.conv2 = Convolution(in_channels=24, out_channels=8, kernel_size=1, spatial_dims=3)
            self.pad = nn.ReflectionPad3d(8)
        if exp == 1:
            self.transp_conv1 = Convolution(
                spatial_dims=spatial_dims,
                in_channels=channels[0],
                out_channels=channels[1],
                strides=4,
                kernel_size=4,
                norm="instance",
                dropout=0.0,
                bias=False,
                conv_only=True,
                is_transposed=True,
                padding=get_padding(4, 4),
                output_padding=get_output_padding(4, 4, get_padding(4, 4)),
            )
            self.transp_conv2 = Convolution(
                spatial_dims=spatial_dims,
                in_channels=channels[1],
                out_channels=channels[2],
                strides=4,
                kernel_size=4,
                norm="instance",
                dropout=0.0,
                bias=False,
                conv_only=True,
                is_transposed=True,
                padding=get_padding(4, 4),
                output_padding=get_output_padding(4, 4, get_padding(4, 4)),
            )
            self.transp_conv3 = Convolution(
                spatial_dims=spatial_dims,
                in_channels=8,
                out_channels=1,
                strides=2,
                kernel_size=2,
                norm="instance",
                dropout=0.0,
                bias=False,
                conv_only=True,
                is_transposed=True,
                padding=get_padding(2, 2),
                output_padding=get_output_padding(2, 2, get_padding(2, 2)),
            )
            self.swin = BasicLayer(dim=8,
                                   depth=2,
                                   num_heads=2,
                                   window_size=(7, 7, 7),
                                   drop_path=[0.0, 0.0],
                                   mlp_ratio=4.0,
                                   qkv_bias=False,
                                   drop=0.0,
                                   attn_drop=0.0,
                                   norm_layer=nn.LayerNorm,
                                   downsample=None,
                                   use_checkpoint=False, )
            self.pad = nn.ReflectionPad3d(8)
            self.conv1 = Convolution(in_channels=48, out_channels=24, kernel_size=1, spatial_dims=3)
            self.conv2 = Convolution(in_channels=24, out_channels=8, kernel_size=1, spatial_dims=3)
        if exp == 2:
            assert len(channels) == 4, "len(channels) == 4"
            self.conv1 = Convolution(in_channels=channels[0], out_channels=channels[1], kernel_size=1, spatial_dims=3)
            self.conv2 = Convolution(in_channels=channels[1], out_channels=channels[2], kernel_size=1, spatial_dims=3)
            self.conv3 = Convolution(in_channels=channels[2], out_channels=channels[3], kernel_size=1, spatial_dims=3)
            self.swin = BasicLayer(dim=8,
                                   depth=2,
                                   num_heads=2,
                                   window_size=(7, 7, 7),
                                   drop_path=[0.0, 0.0],
                                   mlp_ratio=4.0,
                                   qkv_bias=False,
                                   drop=0.0,
                                   attn_drop=0.0,
                                   norm_layer=nn.LayerNorm,
                                   downsample=None,
                                   use_checkpoint=False, )
            self.transp_conv = Convolution(
                spatial_dims=channels[3],
                in_channels=channels[1],
                out_channels=1,
                strides=4,
                kernel_size=4,
                norm="instance",
                dropout=0.0,
                bias=False,
                conv_only=True,
                is_transposed=True,
                padding=get_padding(4, 4),
                output_padding=get_output_padding(4, 4, get_padding(4, 4)),
            )

    def forward(self, x1, x2):
        if self.exp == 0:
            "Custom experiment"
            x11 = self.transp_conv1(x1)
            x12 = self.transp_conv2(x11)
            # x13 = self.pad(x12)
            x13 = x12
            t1 = self.conv1(torch.cat([x2, x13], dim=1))
            t2 = self.conv2(t1)
            t3 = self.swin(t2)
            t4 = self.transp_conv3(t3)
            return t4
        if self.exp == 1:
            "Ablation experiment 1 :without inn"
            x = torch.cat([x1, x2], dim=1)
            x1 = self.transp_conv1(x)
            x2 = self.transp_conv2(x1)
            x3 = self.pad(x2)
            x4 = self.conv1(x3)
            x5 = self.conv2(x4)
            x6 = self.swin(x5)
            x7 = self.transp_conv3(x6)
            return x7
        if self.exp == 2:
            "Ablation experiment 2 :without cnn"
            x = torch.cat([x1, x2], dim=1)
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            x4 = self.swin(x3)
            x5 = self.transp_conv(x4)



# net = Segmentation_Decoder().to("cuda")
# summary(net, [(64, 6, 6, 2), (32, 112, 112, 48)])
