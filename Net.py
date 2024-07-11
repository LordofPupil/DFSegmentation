import torch
from torchsummary import summary

from Model.Encoder import Segmentation_Swin_Encoder, LW_Feature_Extractor, HF_Feature_Extractor
from Model.Fusion import LF_Fusion, HF_Fusion, FSAS
from Model.Decoder import Segmentation_Decoder, FSA
from torch import nn
from collections.abc import Sequence


class Net_Seg(nn.Module):
    def __init__(self,
                 in_channel: int = 1,
                 spatial_dim: int = 3,
                 patch_size: Sequence[int] = (2, 2, 2),
                 embed_dim=16,
                 num_swin_head: Sequence[int] = (2, 2),
                 num_swin_block=2,
                 swin_depth: Sequence[int] = (1, 1),
                 inn_layers: int = 3,
                 conv_channels: Sequence[int] = (16, 24, 32, 64),
                 conv_strides: Sequence[int] = (1, 1, 1),
                 conv_kernel_sizes: Sequence[int] = (3, 3, 3),
                 conv_numblocks: int = 3,
                 conv_act=("prelu", {"init": 0.2}),
                 conv_dropout: float = 0.2,
                 inn_blocktype="MN"
                 ):
        """
        :param in_channel:
        :param in_channel:
        :param spatial_dim:
        :param patch_size:
        :param embed_dim:
        :param num_head:
        :param num_swin_block:
        :param swin_depth:
        :param inn_layers:
        :param conv_channels:
        :param conv_strides:
        :param conv_kernel_sizes:
        :param conv_numblocks:
        :param conv_act:
        :param conv_dropout:
        :param inn_blocktype:
        """
        super(Net_Seg, self).__init__()
        assert conv_channels[0] == embed_dim, "embed_dim must be equal to conv_channels[0]"
        self.swin = Segmentation_Swin_Encoder(in_channel=in_channel,
                                              spatial_dim=spatial_dim,
                                              patch_size=patch_size,
                                              embed_dim=embed_dim,
                                              num_head=num_swin_head,
                                              num_block=num_swin_block,
                                              depth=swin_depth,
                                              )
        self.lw1 = LW_Feature_Extractor(channels=conv_channels,
                                        spatial_dims=spatial_dim,
                                        strides=conv_strides,
                                        kernel_size=conv_kernel_sizes,
                                        num_block=conv_numblocks,
                                        act=conv_act,
                                        dropout=conv_dropout
                                        )
        self.hf1 = HF_Feature_Extractor(num_layers=inn_layers,
                                        inp_dim=embed_dim // 2,
                                        oup_dim=embed_dim // 2,
                                        blocktype=inn_blocktype)
        self.lw2 = LW_Feature_Extractor(channels=conv_channels,
                                        spatial_dims=spatial_dim,
                                        strides=conv_strides,
                                        kernel_size=conv_kernel_sizes,
                                        num_block=conv_numblocks,
                                        act=conv_act,
                                        dropout=conv_dropout
                                        )
        self.hf2 = HF_Feature_Extractor(num_layers=inn_layers,
                                        inp_dim=embed_dim // 2,
                                        oup_dim=embed_dim // 2,
                                        blocktype=inn_blocktype)
        self.lf_funsion= FSAS(dim=64, is_high=False)
        self.hf_funsion = FSAS(dim=32, is_high=True)
        self.ff1 = FSA()
        self.ff2 = FSA()
        self.decoder = Segmentation_Decoder()

    def forward(self, x1, x2):
        x1 = self.swin(x1)
        x2 = self.swin(x2)
        # print(x1.shape, x2.shape)
        x11 = self.lw1(x1)
        x12 = self.hf1(x1)
        x11, x12 = self.ff1(x11, x12)
        print(x11.shape, x12.shape)
        x21 = self.lw2(x2)
        x22 = self.hf2(x2)
        x21, x22 = self.ff2(x21, x22)
        print(x21.shape, x22.shape)
        lf = FSAS(torch.cat([x11, x21], dim=1))
        hf = FSAS(torch.cat([x12, x22], dim=1))
        print(lf.shape, hf.shape)
        out = self.decoder(lf, hf)
        return out, x11, x21, x12, x22


# net = Net_Seg(in_channel=1).to("cuda")
# summary(net, [(1, 224, 224, 96), (1, 224, 224, 96)])










