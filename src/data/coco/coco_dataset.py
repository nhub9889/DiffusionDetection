import torch
import torch.utils.data
import torchvision
from torchvision import tv_tensors
from .coco_utils import ConvertCocoPolysToMask
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, remap_mscoco_category=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask()
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):

        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]

        target = {'image_id': image_id, 'annotations': target}

        img, target = self.prepare(img, target)

        if 'boxes' in target:
            w, h = img.size
            target['boxes'] = tv_tensors.BoundingBoxes(
                target['boxes'],
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w)
            )

        if 'masks' in target:
            target['masks'] = tv_tensors.Mask(target['masks'])

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target