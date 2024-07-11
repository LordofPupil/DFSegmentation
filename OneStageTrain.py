import os
import time

import torch
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric, compute_hausdorff_distance, compute_average_surface_distance
from monai.transforms import Compose, Activations, AsDiscrete
from torch.utils.data import SubsetRandomSampler
from torchsummary import summary
import argparse
from easydict import EasyDict as edict
from monai import transforms
import datamodule
import DataProcessing
from DataProcessing import HuaxiDataset
from Net import Net_Seg
from datamodule import ResolveModalityDataset
from loss import cc, fcc
from utils import write_pickle, write_to_file, imgshow1


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    device = torch.device("cuda:0")
    opt = edict()
    net = Net_Seg().to(device)
    if args.retrain:
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    DCELoss = DiceCELoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    DICELoss = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    Dice_Metric = DiceMetric(reduction="mean")
    # HD_Metric = HausdorffDistanceMetric(reduction="mean", percentile=95)
    # AVD_Metric = SurfaceDistanceMetric(reduction="mean")
    opt.best_metric = 0.0
    opt.DiceLossList = []
    opt.TrainLossList = []
    opt.DiceMetricList = []
    opt.HDMetricList = []
    opt.AsdMetricList = []
    dataset = HuaxiDataset(dictionary=args.datadict, data_transform=args.data_transform)
    # dataset = ResolveModalityDataset(dictionary=args.datadict, data_transform=args.data_transform)
    dataset_size = len(dataset)
    split = int(0.8 * dataset_size)

    torch.manual_seed(args.seed)

    indices = torch.randperm(dataset_size)

    train_indices = indices[:split]
    test_indices = indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.num_workers)

    num_epochs = args.num_epochs
    coeff_decomp = args.coeff_decomp
    val_interval = args.val_interval
    hd_metric = 0.
    asd_metric = 0.
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss_sum = 0
        dice_loss_sum = 0
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        step = 0
        net.train()
        for data_m1, data_m2, label in train_loader:
            data_m1, data_m2, label = data_m1.to(device), data_m2.cuda(device), label.cuda(device)
            optimizer.zero_grad()
            step = step + 1
            output, feature1_B, feature2_B, feature1_D, feature2_D = net(data_m1, data_m2)
            cc_loss_B = cc(feature1_B, feature2_B)
            cc_loss_D = cc(feature1_D, feature2_D)
            dice_loss = DCELoss(output, label)
            dice_m = DICELoss(output, label)
            loss_decomp = ((cc_loss_D ** 2) / (0.01 + cc_loss_B))
            loss_fdecomp = -(fcc(feature1_B, feature1_D)+fcc(feature2_B, feature2_D))
            loss = coeff_decomp * loss_decomp + dice_loss+ 0.1*loss_fdecomp
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()
            epoch_loss_sum = epoch_loss_sum + epoch_loss
            dice = dice_m.item()
            dice_loss_sum = dice_loss_sum + (1 - dice)
            print(
                f"{step}/{len(train_loader)} in epoch{epoch}"
                f", train_loss: {loss.item():.4f}"
                f",Dice:{1 - dice:.4f}，loss_decomp：{loss_decomp:.4f}"
            )

        opt.DiceLossList.append(dice_loss_sum / step)
        opt.TrainLossList.append(epoch_loss_sum / step)
        opt.HDMetricList.append(hd_metric/ (step+1))
        opt.AsdMetricList.append(asd_metric/(step + 1))
        scheduler.step()

        if (epoch + 1) % val_interval == 0:
            step_val = 0
            net.eval()
            with torch.no_grad():
                for data_m1, data_m2, val_label in test_loader:
                    step_val = step_val + 1
                    print(f"validation step {step_val}")
                    data_m1, data_m2, val_labels = data_m1.cuda(), data_m2.cuda(), val_label.cuda()
                    output, feature1_B, feature2_B, feature1_D, feature2_D = net(data_m1, data_m2)
                    val_outputs = [post_trans(i) for i in decollate_batch(output)]
                    val_outputs = torch.unsqueeze(val_outputs[0], dim=0)
                    hd_metric = hd_metric + compute_hausdorff_distance(y_pred=val_outputs, y=label, spacing=1.0,percentile=95).item()
                    asd_metric = asd_metric + compute_average_surface_distance(y_pred=val_outputs,y=label,spacing=1.0).item()
                    Dice_Metric(y_pred=val_outputs, y=val_labels)

            metric = Dice_Metric.aggregate().item()
            opt.DiceMetricList.append(metric)
            Dice_Metric.reset()
            if metric > opt.best_metric:
                opt.best_metric = metric
                model = {
                    'net': net.state_dict()
                }
                torch.save(
                    model,
                    "modelsae/best_metric_model_23.pth"
                )
            checkpoint = {
                "epoch": epoch,
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(f"checkpoint4/DFSegmentation{epoch}"))
            # print("checkpoint file save correctly")
            showcontent = (
                    "######################################################################\n" +
                    f"epoch{epoch} is over ,epoch time is {(time.time() - epoch_start):.4f}\n" +
                    f"average epoch train loss is {(epoch_loss_sum / step):.4f}\n" +
                    f"average epoch train dice is {(dice_loss_sum / step):.4f}\n" +
                    f"validation dice is {(metric):.4f}\n" +
                    f"validation hd is {(hd_metric/(step+1)):.4f}\n" +
                    f"validation asd is {(asd_metric/(step+1)):.4f}\n" +
                    f"best dice is {(opt.best_metric):.4f}"
            )
            hd_metric = 0
            asd_metric = 0
            print(showcontent)
            # write_pickle(opt, "optpkl3")
            # write_to_file("example4.txt", showcontent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadict', default=DataProcessing.datadict, help='pathdict to dataset')
    # parser.add_argument('--datadict', default=datamodule.datadict, help='pathdict to dataset')
    parser.add_argument('--num_epochs', type=int, default=300, help='max epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--val_interval', type=int, default=1, help='Perform validation every few epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--data_transform', default=DataProcessing.h_transform, help='transform')
    # parser.add_argument('--data_transform', default=datamodule.data_transform, help='transform')
    parser.add_argument('--coeff_decomp', type=float, default=1.0, help='coeff_decomp')
    parser.add_argument('--retrain', type=bool, default=True, help='is retrain')
    parser.add_argument('--checkpoint', type=str, default="checkpoint3/DFSegmentation12", help='is retrain')
    parser.add_argument('--exp', type=int, default=0, help='defferent experiment')
    args = parser.parse_args()
    main(args)


