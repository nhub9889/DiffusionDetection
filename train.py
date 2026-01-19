import torch
import torch.optim as optim
import time
import datetime
import os
import argparse
import torchvision.transforms as T

from src.core.model import DiffusionDetModel
from src.data.dataloader import DataLoader
from src.data.coco.coco_dataset import CocoDetection


def get_transform(train=True):
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.Resize((640, 640), antialias=True))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    header = f'Epoch: [{epoch}]'
    total_loss = 0
    start_time = time.time()

    for i, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(losses):
            print(f"Loss is {losses}, stopping training")
            print(loss_dict)
            continue

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += losses.item()

        if i % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"{header} Step [{i}/{len(data_loader)}] "
                  f"Loss: {losses.item():.4f} (Avg: {total_loss / (i + 1):.4f}) "
                  f"LR: {lr:.6f}")

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}. Time: {time.time() - start_time:.2f}s")
    return avg_loss

def main(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    print("Creating model...")
    model = DiffusionDetModel(args.config)
    model.to(device)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
        else:
            print(f"No checkpoint found at {args.resume}")

    print("Loading data...")
    dataset_train = CocoDetection(
        img_folder=f"{args.data_path}/train2017",
        ann_file=f"{args.data_path}/annotations/instances_train2017.json",
        transforms=get_transform(train=True),
        return_masks=False
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    print(f"Data loaded: {len(dataset_train)} images.")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("Start training...")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=args.print_freq)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
            torch.save({'model': model.state_dict()}, os.path.join(args.output_dir, "latest.pth"))
            print(f"Saved checkpoint to {checkpoint_path}")

    total_time = time.time() - start_time
    print(f"Training time: {str(datetime.timedelta(seconds=int(total_time)))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DiffusionDet")
    # ... (Các arguments giữ nguyên)
    parser.add_argument('--config', default='resnet18.yaml', help='path to yaml config file')
    parser.add_argument('--data-path', default='/kaggle/input/coco-2017-dataset/coco2017',
                        help='path to coco dataset root')
    parser.add_argument('--output-dir', default='./output', help='path to save checkpoints')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--lr', default=2.5e-5, type=float)
    parser.add_argument('--lr-backbone', default=1e-5, type=float)

    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--print-freq', default=50, type=int)

    args = parser.parse_args()
    main(args)