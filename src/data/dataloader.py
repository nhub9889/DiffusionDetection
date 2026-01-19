import torch
import torch.utils.data as data
class DataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        super().__init__(
            dataset,
            batch_size= batch_size,
            num_workers= num_workers,
            shuffle= shuffle,
            collate_fn= diffusion_collate_fn
        )
def diffusion_collate_fn(batch):
    images = []
    batch_boxes = []
    batch_classes = []
    batch_masks = []

    MAX_OBJECTS = 100
    for img, target in batch:
        images.append(img)

        boxes = target['boxes']
        labels = target['labels']

        padded_boxes = torch.zeros((MAX_OBJECTS, 4), dtype=torch.float32)
        padded_labels = torch.zeros((MAX_OBJECTS), dtype=torch.long)
        mask = torch.zeros((MAX_OBJECTS), dtype=torch.bool)

        num_objs = min(len(boxes), MAX_OBJECTS)
        if num_objs > 0:
            padded_boxes[:num_objs] = boxes[:num_objs]
            padded_labels[:num_objs] = labels[:num_objs]
            mask[:num_objs] = True
        batch_boxes.append(padded_boxes)
        batch_classes.append(padded_labels)
        batch_masks.append(mask)

        images = torch.stack(images, dim= 0)
        gt_boxes = torch.stack(batch_boxes, dim= 0)
        gt_labels = torch.stack(batch_classes, dim= 0)
        gt_masks = torch.stack(batch_masks, dim= 0)

        return{
            "images": images,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "gt_masks": gt_masks,
        }