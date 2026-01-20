import os
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

COCO_LABEL_MAP = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20,
    22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
    35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
    46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50,
    56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60,
    67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70,
    80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80
}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if not polygons:  # Handle empty segmentation
            mask = torch.zeros((height, width), dtype=torch.uint8)
        else:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        boxes[:, 2:] += boxes[:, :2]

        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]

        if self.remap_mscoco_category:
            classes = [COCO_LABEL_MAP[c] for c in classes]

        classes = torch.tensor(classes, dtype=torch.int64)
        if self.remap_mscoco_category:
            classes -= 1

        if self.return_masks and "segmentation" in anno[0]:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        else:
            masks = None

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        if masks is not None:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target_new = {}
        target_new["boxes"] = boxes
        target_new["labels"] = classes
        if masks is not None:
            target_new["masks"] = masks
        target_new["image_id"] = image_id
        if keypoints is not None:
            target_new["keypoints"] = keypoints

        area = torch.tensor([obj["area"] for obj in anno])
        target_new["area"] = area[keep]

        target_new["iscrowd"] = torch.tensor([obj["iscrowd"] for obj in anno])[keep]
        target_new["orig_size"] = torch.tensor([h, w])
        target_new["size"] = torch.tensor([h, w])

        return image, target_new