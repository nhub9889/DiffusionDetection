import copy
import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision.ops.roi_align as roi_align

from torchvision.ops import roi_align


class MultiScaleRoIAlign(nn.Module):
    def __init__(self, output_size, feature_strides, image_size_base=224):
        super().__init__()
        self.output_size = output_size
        self.feature_strides = feature_strides
        self.image_size_base = image_size_base

    def forward(self, features, boxes):

        if isinstance(boxes, torch.Tensor):
            boxes = [b for b in boxes]

        num_images = len(boxes)
        num_inputs = len(features)

        assert num_inputs == len(self.feature_strides)

        final_output = torch.zeros(
            (sum(len(b) for b in boxes), features[0].size(1), *self.output_size),
            dtype=features[0].dtype,
            device=features[0].device
        )

        all_boxes = torch.cat(boxes, dim=0)

        areas = (all_boxes[:, 2] - all_boxes[:, 0]) * (all_boxes[:, 3] - all_boxes[:, 1])
        scale = torch.sqrt(areas)

        target_lvls = torch.floor(4 + torch.log2(scale / self.image_size_base + 1e-6))

        min_level = 2
        max_level = min_level + num_inputs - 1

        target_lvls = torch.clamp(target_lvls, min=min_level, max=max_level)
        target_lvls = target_lvls - min_level
        current_idx = 0
        box_indices_list = []
        for level_idx, (feat, stride) in enumerate(zip(features, self.feature_strides)):
            idx_in_level = torch.where(target_lvls == level_idx)[0]

            if len(idx_in_level) == 0:
                continue

            boxes_in_level = all_boxes[idx_in_level]
            boxes_per_image = [len(b) for b in boxes]
            image_indices = torch.cat([
                torch.full((count,), i, dtype=torch.int64, device=all_boxes.device)
                for i, count in enumerate(boxes_per_image)
            ])

            imgs_idx_lvl = image_indices[idx_in_level]
            rois_with_batch_idx = torch.cat([
                imgs_idx_lvl.unsqueeze(1).float(),
                boxes_in_level
            ], dim=1)  # (M, 5)

            roi_feats_level = roi_align(
                feat,
                rois_with_batch_idx,
                output_size=self.output_size,
                spatial_scale=1.0 / stride,
                sampling_ratio=2,
                aligned=True
            )

            final_output[idx_in_level] = roi_feats_level

        return final_output
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class DynamicConv(nn.Module):
    def __init__(self, hidden_dim=256, dim_dynamic=64, num_dynamic=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_dynamic = dim_dynamic
        self.num_dynamic = num_dynamic
        self.num_params = hidden_dim * dim_dynamic

        # Sinh ra weights động từ feature của proposal
        self.dynamic_layer = nn.Linear(hidden_dim, num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(dim_dynamic)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU(inplace=True)

        # Output layer
        pooler_resolution = 7
        num_output = hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, pro_features, roi_features):
        """
        pro_features: (1, Total_Boxes, C) -> Dùng để sinh weights
        roi_features: (49, Total_Boxes, C) -> Feature ảnh từ RoIAlign (đã flatten spatial)
        """
        # Sinh parameters
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)  # (Total_Boxes, 1, num_params*2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        # Thực hiện Dynamic Interaction
        # roi_features: (Total_Boxes, 49, C)
        features = roi_features.permute(1, 0, 2)

        # Layer 1
        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        # Layer 2
        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        # Flatten và Project về C
        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


# --- 2. RCNNHead (Một bước tinh chỉnh) ---

class RCNNHead(nn.Module):
    def __init__(self, d_model, num_classes, fpn_strides,dim_feedforward=2048,
                 num_cls=1, num_reg=3, num_dynamic=2, dim_dynamic=64,
                 bbox_weights=(2.0, 2.0, 1.0, 1.0), scale_clamp=math.log(1000.0 / 16)):
        super().__init__()
        self.d_model = d_model
        self.bbox_weights = bbox_weights
        self.scale_clamp = scale_clamp

        # Dynamic Interaction Components
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1)
        self.inst_interact = DynamicConv(d_model, dim_dynamic, num_dynamic)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = F.relu

        # Time Modulation
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2))

        # Prediction Heads
        self.cls_module = self._make_fc_layers(d_model, num_cls)
        self.reg_module = self._make_fc_layers(d_model, num_reg)
        self.box_pooler = MultiScaleRoIAlign(
            output_size=(7, 7),
            feature_strides=fpn_strides
        )
        self.class_logits = nn.Linear(d_model, num_classes)
        self.bboxes_delta = nn.Linear(d_model, 4)

    def _make_fc_layers(self, d_model, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(d_model, d_model, False))
            layers.append(nn.LayerNorm(d_model))
            layers.append(nn.ReLU(inplace=True))
        return nn.ModuleList(layers)

    def forward(self, feature_map, bboxes, pro_features, time_emb, pooler_resolution, spatial_scale):
        N, nr_boxes = bboxes.shape[:2]

        box_list = [bboxes[i] for i in range(N)]

        roi_features = self.box_pooler(feature_map, bboxes)

        if pro_features is None:
            pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1)

        roi_features_flat = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)

        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = self.norm1(pro_features + pro_features2)
        pro_features_flat_input = pro_features.permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features_flat_input, roi_features_flat)
        pro_features2 = pro_features2.view(N, nr_boxes, self.d_model).permute(1, 0, 2)  # Trả về (nr_boxes, Batch, C)
        pro_features = self.norm2(pro_features + pro_features2)

        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(pro_features))))
        obj_features = self.norm3(pro_features + obj_features2)

        # 5. Time Modulation (Scale & Shift feature theo thời gian t)
        fc_feature = obj_features.permute(1, 0, 2).reshape(N * nr_boxes, -1)

        scale_shift = self.block_time_mlp(time_emb)  # (Batch, C*2)
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)

        fc_feature = fc_feature * (scale + 1) + shift

        # 6. Predict Class & Box Delta
        cls_feat = fc_feature
        reg_feat = fc_feature

        for layer in self.cls_module: cls_feat = layer(cls_feat)
        for layer in self.reg_module: reg_feat = layer(reg_feat)

        class_logits = self.class_logits(cls_feat)
        bboxes_deltas = self.bboxes_delta(reg_feat)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.reshape(-1, 4))

        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features.permute(1, 0, 2)

    def apply_deltas(self, deltas, boxes):
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
        return pred_boxes

class DynamicHead(nn.Module):
    def __init__(self, num_classes=80, hidden_dim=256, dim_feedforward=2048,
                 num_heads=6, num_cls=1, num_reg=3, num_dynamic=2,
                 fpn_strides=[4, 8, 16, 32], dim_dynamic=64):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        time_dim = hidden_dim * 4
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.head_series = nn.ModuleList([
            RCNNHead(hidden_dim, num_classes, dim_feedforward,
                     num_cls=num_cls, num_reg=num_reg,
                     num_dynamic=num_dynamic, dim_dynamic=dim_dynamic)
            for _ in range(num_heads)
        ])

        self.pooler_resolution = 7
        self.pooler_scale = 1.0 / 32.0

        self._reset_parameters()
        self.fpn_strides = fpn_strides

        # Khởi tạo MultiScaleRoIAlign
        self.box_pooler = MultiScaleRoIAlign(
            output_size=(7, 7),
            feature_strides=fpn_strides
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, init_bboxes, t, init_features=None):

        time_emb = self.time_mlp(t)

        feature_map = features

        inter_class_logits = []
        inter_pred_bboxes = []

        bs, num_boxes = init_bboxes.shape[:2]
        bboxes = init_bboxes

        if init_features is not None:
            proposal_features = init_features.clone()
        else:
            proposal_features = None

        for rcnn_head in self.head_series:
            class_logits, pred_bboxes, proposal_features = rcnn_head(
                feature_map, bboxes, proposal_features, time_emb, self.box_pooler
            )

            inter_class_logits.append(class_logits)
            inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)