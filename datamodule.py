from monai.apps import DecathlonDataset
import glob
import os
from utils import *
import torch
from monai import transforms
from monai.transforms import MapTransform, LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd, Spacingd, \
    RandSpatialCropd, RandFlipd, NormalizeIntensityd, RandShiftIntensityd, RandScaleIntensityd, CenterSpatialCropd
from monai.data import Dataset, DataLoader
import torchvision

images_dir = "BraTs2016\Task01_BrainTumour\imagesTr"
labels_dir = "BraTs2016\Task01_BrainTumour\labelsTr"

images = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
labels = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
datadict = [{'image': image, 'label': label}
            for image, label in zip(images, labels)]


# for i in range(len(datadict)):
#     print(datadict[i])

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            result.append(d[key] == 2)
            d[key] = torch.stack(result).float()
        return d


class ConvertToTwoModality(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key][1, :, :, :])
            result.append(d[key][2, :, :, :])
            d[key] = torch.stack(result).float()
        return d


class ResolveModality(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result1 = d[key][0, :, :, :]
            result2 = d[key][1, :, :, :]
        return result1, result2


data_transform = transforms.Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys="image"),
    EnsureTyped(keys=["image", "label"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    ConvertToTwoModality(keys="image"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "nearest"),
    ),
    CenterSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 96]),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
])


class addchannelfirst(MapTransform):
    def __call__(self, data):
        data = torch.unsqueeze(data, 0)
        return data


class ResolveModalityDataset(Dataset):

    def __init__(self, dictionary, data_transform=None):
        self.dictionary = dictionary
        self.data_transform = data_transform
        self.tem_tranform = transforms.Compose([
            ResolveModality(keys="image")
        ])

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        data_path = self.dictionary[idx]
        data = self.data_transform(data_path)
        data_m1, data_m2 = self.tem_tranform(data)
        ecf = transforms.Compose([
            addchannelfirst(keys="image")
        ])
        data_m1, data_m2 = ecf(data_m1), ecf(data_m2)
        return data_m1, data_m2, data["label"]

# train_ds = ResolveModalityDataset(dictionary=datadict, data_transform=data_transform)
# loader = DataLoader(train_ds,
#                     batch_size=1)
# for data_m1, data_m2, label in loader:
#     imgshow1(data_m1, data_m2, label, 0)
