import torch
import torch.nn as nn
import torchvision
from torchvision import tv_tensors
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image
from typing import Any, Dict, List, Optional

torchvision.disable_beta_transforms_warning()

class Compose(T.Compose):
    def __init__(self, transforms: list) -> None:
        super().__init__(transforms=transforms)

class EmptyTransform(T.Transform):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs

class PadToSize(T.Pad):
    def __init__(self, spatial_size, fill=0, padding_mode='constant') -> None:
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        self.spatial_size = spatial_size
        super().__init__(0, fill, padding_mode)

    def forward(self, *inputs: Any) -> Any:
        flat_inputs = torch.utils._pytree.tree_flatten(inputs)[0]
        img = flat_inputs[0]
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        elif isinstance(img, Image.Image):
            w, h = img.size
        else:
             h, w = img.shape[-2:]

        pad_h = max(0, self.spatial_size[0] - h)
        pad_w = max(0, self.spatial_size[1] - w)

        padding = [0, 0, pad_w, pad_h]

        return F.pad(inputs, padding=padding, fill=self.fill, padding_mode=self.padding_mode)

class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5,
                 max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40,
                 p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)

class ConvertBox(T.Transform):
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            'xyxy': tv_tensors.BoundingBoxFormat.XYXY,
            'cxcywh': tv_tensors.BoundingBoxFormat.CXCYWH
        }

    def forward(self, *inputs: Any) -> Any:
        return super().forward(*inputs)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, tv_tensors.BoundingBoxes):
            if self.out_fmt:
                # Convert format
                in_fmt = inpt.format
                target_fmt = self.data_fmt[self.out_fmt]
                inpt = F.convert_bounding_box_format(inpt, new_format=target_fmt)

            if self.normalize:
                inpt = F.normalize_bounding_box(inpt, canvas_size=inpt.canvas_size)
                
        return inpt