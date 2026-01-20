import torch
import torch.nn as nn
import math
import yaml

from .encoder import Encoder
from .head import DynamicHead
from .detector import SetCriterionLight, HungarianMatcherLight


class DiffusionDetModel(nn.Module):
    def __init__(self, cfg_path):
        super().__init__()
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.encoder = Encoder(self.cfg['model'])
        self.fpn_strides = [4, 8, 16, 32]

        head_cfg = self.cfg['head']
        self.head = DynamicHead(
            num_classes=head_cfg['num_classes'],
            hidden_dim=head_cfg['hidden_dim'],
            dim_feedforward=head_cfg['dim_feedforward'],
            num_heads=head_cfg['num_heads'],
            num_cls=head_cfg['num_cls'],
            num_reg=head_cfg['num_reg'],
            num_dynamic=head_cfg['num_dynamic'],
            dim_dynamic=head_cfg['dim_dynamic'],
            fpn_strides=self.fpn_strides
        )

        matcher_cfg = self.cfg['matcher']
        loss_cfg = self.cfg['loss']

        self.matcher = HungarianMatcherLight(
            cfg=self.cfg,
            cost_class=matcher_cfg['cost_class'],
            cost_bbox=matcher_cfg['cost_bbox'],
            cost_giou=matcher_cfg['cost_giou'],
            use_focal=matcher_cfg['use_focal']
        )
        # Gán thêm params đặc biệt
        self.matcher.ota_k = matcher_cfg['ota_k']
        self.matcher.focal_loss_alpha = matcher_cfg['alpha']
        self.matcher.focal_loss_gamma = matcher_cfg['gamma']

        self.criterion = SetCriterionLight(
            cfg=self.cfg,
            num_classes=head_cfg['num_classes'],
            matcher=self.matcher,
            weight_dict=loss_cfg['weight_dict'],
            eos_coef=loss_cfg['eos_coef'],
            losses=['labels', 'boxes'],
            use_focal=matcher_cfg['use_focal']
        )

    def forward(self, images, targets=None):
        """
        images: Tensor (Batch, C, H, W)
        targets: List[Dict] (chứa 'boxes', 'labels') - Chỉ cần khi training
        """

        features = self.encoder(images)

        batch_size = images.shape[0]

        if self.training:
            t = torch.randint(0, 1000, (batch_size,), device=images.device)
            rand_boxes = torch.rand(batch_size, 100, 4, device=images.device)
            cx, cy, w, h = rand_boxes.unbind(-1)

            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h

            init_bboxes = torch.stack([x1, y1, x2, y2], dim=-1) * 640.0
            init_bboxes = init_bboxes.clamp(min=0.0, max=640.0)
            outputs_class, outputs_coords = self.head(features, init_bboxes, t)

            output_dict = {
                'pred_logits': outputs_class[-1],
                'pred_boxes': outputs_coords[-1]
            }

            loss_dict = self.criterion(output_dict, targets)
            return loss_dict

        else:
            return self.inference(features)

    @torch.no_grad()
    def inference(self, features):
        batch_size = features[0].shape[0]
        device = features[0].device
        num_proposals = self.cfg['head']['num_proposals']

        rand_boxes = torch.rand(batch_size, num_proposals, 4, device=device)
        cx, cy, w, h = rand_boxes.unbind(-1)

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        current_boxes = torch.stack([x1, y1, x2, y2], dim=-1) * 640.0
        current_boxes = current_boxes.clamp(min=0.0, max=640.0)
        steps = [0]

        results = []
        for t in steps:
            time_tensor = torch.full((batch_size,), t, device=device)

            pred_logits, pred_boxes = self.head(features, current_boxes, time_tensor)
            current_boxes = pred_boxes[-1]
            final_scores = pred_logits[-1].sigmoid()

            results.append({'boxes': current_boxes, 'scores': final_scores})

        return results