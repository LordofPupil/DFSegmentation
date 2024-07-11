import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W, D) image; torch.float32 [0.,1.]."""
    N, C, _, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(
        torch.sum(img2 ** 2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()


def tensor2freq(self, x):
    # Adjust for 3D images: crop volume patches
    patch_factor = self.patch_factor
    _, _, d, h, w = x.shape
    assert d % patch_factor == 0 and h % patch_factor == 0 and w % patch_factor == 0, (
        'Patch factor should be divisible by volume depth, height, and width')
    patch_list = []
    patch_d = d // patch_factor
    patch_h = h // patch_factor
    patch_w = w // patch_factor
    for i in range(patch_factor):
        for j in range(patch_factor):
            for k in range(patch_factor):
                patch_list.append(x[:, :, i * patch_d:(i + 1) * patch_d, j * patch_h:(j + 1) * patch_h,
                                  k * patch_w:(k + 1) * patch_w])

    # Stack to patch tensor and perform 3D FFT
    y = torch.stack(patch_list, 1)

    freq = torch.fft.fftn(y, norm='ortho')
    freq = torch.stack([freq.real, freq.imag], -1)
    return freq

def fcc(recon_freq, real_freq):

    # frequency distance using (squared) Euclidean distance
    tmp = (recon_freq - real_freq) ** 2
    freq_distance = tmp[..., 0] + tmp[..., 1]

    # dynamic spectrum weighting (Hadamard product)
    loss = freq_distance
    return torch.mean(loss)