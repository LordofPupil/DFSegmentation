import glob
from monai.apps import DecathlonDataset
import glob
import os
from utils import *
import torch
from monai import transforms
from monai.transforms import MapTransform, LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd, Spacingd, \
    RandSpatialCropd, RandFlipd, NormalizeIntensityd, RandShiftIntensityd, RandScaleIntensityd, CenterSpatialCropd, \
    SpatialPadd
from monai.data import Dataset, DataLoader
import torchvision

DWI_dir = "All/images_dwi"
T2_dir = "All/images_t2"
tumor_dir = "All/labels_tumor"
outlnr_dir = "All/labels_outlnr"

dwi = sorted(glob.glob(os.path.join(DWI_dir, "*.nii.gz")))
t2 = sorted(glob.glob(os.path.join(T2_dir, "*.nii.gz")))
tumor = sorted(glob.glob(os.path.join(tumor_dir, "*.nii.gz")))
outlnr = sorted(glob.glob(os.path.join(outlnr_dir, "*.nii.gz")))

datadict = [{'dwi': d, 't2': t, 'tumor': r}
            for d, t, r in zip(dwi, t2, tumor)]

# print(len(datadict))
# for i in range(len(datadict)):
#     print('#####################\n')
#     print(f"dwi: {datadict[i]['dwi']}\n")
#     print(f"t2: {datadict[i]['t2']}\n")
#     print(f"tumor: {datadict[i]['tumor']}\n")
#     print(f"outlnr: {datadict[i]['outlnr']}\n")


class showshape(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            print(key, d[key].shape)
        return d

h_transform = transforms.Compose([
    LoadImaged(keys=['dwi', 't2', 'tumor']),
    EnsureChannelFirstd(keys=['dwi', 't2', 'tumor']),
    EnsureTyped(keys=['dwi', 't2', 'tumor']),
    # showshape(keys=['dwi', 't2', 'tumor']),
    Orientationd(keys=['dwi', 't2', 'tumor'], axcodes="RAS"),
    Spacingd(
        keys=['dwi', 't2', 'tumor'],
        pixdim=(1.0, 1.0, 5.0),
        mode=("bilinear", "bilinear", "nearest"),
    ),
    CenterSpatialCropd(keys=['dwi', 't2', 'tumor'], roi_size=[256, 256, 32]),
    SpatialPadd(keys=['dwi', 't2', 'tumor'], spatial_size=[256, 256, 32]),
    # RandFlipd(keys=['dwi', 't2', 'tumor', 'outlnr'], prob=0.5, spatial_axis=0),
    # RandFlipd(keys=['dwi', 't2', 'tumor', 'outlnr'], prob=0.5, spatial_axis=1),
    # RandFlipd(keys=['dwi', 't2', 'tumor', 'outlnr'], prob=0.5, spatial_axis=2),
    NormalizeIntensityd(keys=['dwi', 't2'], nonzero=True, channel_wise=True),
    # RandScaleIntensityd(keys=['dwi', 't2', 'tumor', 'outlnr'], factors=0.1, prob=1.0),
    # RandShiftIntensityd(keys=['dwi', 't2', 'tumor', 'outlnr'], offsets=0.1, prob=1.0),
])


class HuaxiDataset(Dataset):

    def __init__(self, dictionary, data_transform=None, type=0):
        self.dictionary = dictionary
        self.data_transform = data_transform
        self.type = type

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        data_path = self.dictionary[idx]
        # print(data_path)
        data = self.data_transform(data_path)
        # data_m1, data_m2 = self.tem_tranform(data)
        return data['dwi'], data['t2'], data["tumor"]


# train_ds = HuaxiDataset(dictionary=datadict, data_transform=h_transform)
# loader = DataLoader(train_ds,
#                     batch_size=1)
#
# tem = [0, 0, 0]
# h = 0
# for data_m1, data_m2, label in loader:
#     imgshow1(data_m1, data_m2, label, 15)