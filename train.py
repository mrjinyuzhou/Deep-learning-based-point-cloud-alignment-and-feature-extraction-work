import os
import time
import datetime
from typing import Union, List
import torch
import torch.nn as nn
from torch.utils import data
from src import PointNet
from train_utils import train_and_eval
from train_utils import distributed_utils as utils
from Data import POINTDataset
import transforms as T
from torchvision.transforms import functional as F


class PresetTrain:
    def __init__(self, num_samples):
        self.transforms = T.Compose([
            T.Downsample(num_samples=num_samples),
            T.ToTensor(),
            T.Flaten()
        ])

    def __call__(self, point, target):
        return self.transforms(point, target)


class PresetEval:
    def __init__(self, num_samples):
        self.transforms = T.Compose([
            T.Downsample(num_samples=num_samples),
            T.ToTensor(),
            T.Flaten()

        ])

    def __call__(self, point, target):
        return self.transforms(point, target)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = POINTDataset(args.data_path, train=True, transforms=PresetTrain(num_samples=2024))
    val_dataset = POINTDataset(args.data_path, train=False, transforms=PresetEval(num_samples=2024))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=True,
                                        collate_fn=train_dataset.collate_fn)

    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1, 
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      collate_fn=val_dataset.collate_fn)

    model = PointNet()
    model = model.double()
    model.to(device)

    params_group = train_and_eval.get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = train_and_eval.create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=2)
    criterion = nn.MSELoss()
    current_mae = 1.0
    start_time = time.time()
    for epoch in range(args.epochs):
        loss, lr = train_and_eval.train_one_epoch(model, train_data_loader, optimizer, criterion, lr_scheduler, device, epoch, scaler)
        
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            mae = train_and_eval.evaluation(model, val_data_loader, device)
            mae_info = mae.compute()
            with open(results_file, "a") as f:
                write_info = f"[epoch: {epoch}] train_loss: {loss:.4f} lr: {lr:.6f} " \
                             f"MAE: {mae_info:.3f}\n"
                f.write(write_info)

            if current_mae >= mae_info:
                torch.save(save_file, "save_weights/model_best.pth")
        
        if os.path.exists(f"save_weights/model_{epoch-10}.pth"):
            os.remove(f"save_weights/model_{epoch-10}.pth")

        torch.save(save_file, f"save_weights/model_{epoch}.pth")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch u2net training")

    parser.add_argument("--data-path", default="./", help="DATA root")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--epochs", default=360, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--eval-interval", default=10, type=int, help="validation interval default 10 Epochs")

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument("--amp", action='store_true',
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
