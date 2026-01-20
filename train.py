import torch
import torch.optim as optim
import time
import datetime
import os
import argparse

# Import torchvision v2
from torchvision.transforms import v2 as T
from torchvision import tv_tensors

# Import module của bạn
from src.core.model import DiffusionDetModel
from src.data.dataloader import DataLoader
from src.data.coco.coco_dataset import CocoDetection


def get_transform(train=True):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        T.RandomPhotometricDistort() # Có thể bật nếu muốn augment màu

    transforms.append(T.Resize((640, 640), antialias=True))

    transforms.append(T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    transforms.append(T.SanitizeBoundingBoxes())

    return T.Compose(transforms)


def collate_fn(batch):
    images = []
    targets = []

    for img, target in batch:
        images.append(img)

        boxes_xyxy = target['boxes']
        if isinstance(boxes_xyxy, tv_tensors.BoundingBoxes):
            boxes_xyxy = boxes_xyxy.as_subclass(torch.Tensor)

        h, w = img.shape[-2:]
        img_size = torch.tensor([w, h, w, h], dtype=torch.float32)

        if boxes_xyxy.shape[0] > 0:
            cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
            cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
            bw = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0])
            bh = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
            boxes_cxcywh = torch.stack([cx, cy, bw, bh], dim=-1)

            boxes_norm = boxes_cxcywh / img_size
        else:
            boxes_norm = torch.zeros((0, 4), dtype=torch.float32)

        new_target = {
            'labels': target['labels'],
            'boxes': boxes_norm,
            'boxes_xyxy': boxes_xyxy,
            'image_size_xyxy': img_size,
            'image_size_xyxy_tgt': img_size
        }

        targets.append(new_target)

    images = torch.stack(images, dim=0)
    return images, targets


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Tăng clip lên xíu
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

    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint {args.resume}")

    print("Loading data...")
    dataset_train = CocoDetection(
        img_folder=f"{args.data_path}/train2017",
        ann_file=f"{args.data_path}/annotations/instances_train2017.json",
        transforms=get_transform(train=True),
        return_masks=False
    )

    print(f"Dataset size: {len(dataset_train)}")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn  # Dùng collate_fn đơn giản ở trên
    )

    # Optimizer setup
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
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"))

    print(f"Total time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='resnet18.yaml')
    parser.add_argument('--data-path', default='/kaggle/input/coco-2017-dataset/coco2017')
    parser.add_argument('--output-dir', default='./output')
    parser.add_argument('--resume', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--lr', default=2.5e-5, type=float)
    parser.add_argument('--lr-backbone', default=1e-5, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--print-freq', default=50, type=int)

    args = parser.parse_args()
    main(args)