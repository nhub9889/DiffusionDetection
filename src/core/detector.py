import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
import torchvision.ops as ops
from .util import box_ops
from .util.misc import get_world_size, is_dist_avail_and_initialized
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou


class SetCriterionLight(nn.Module):
    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses, use_focal):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal

        if self.use_focal:
            self.focal_loss_alpha = cfg.matcher.alpha
            self.focal_loss_gamma = cfg.matcher.alpha
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        batch_size = len(targets)

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            target_classes_o = targets[batch_idx]["labels"]
            target_classes[batch_idx, valid_query] = target_classes_o[gt_multi_idx]

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        src_logits = src_logits.flatten(0, 1)
        target_classes_onehot = target_classes_onehot.flatten(0, 1)

        if self.use_focal:
            cls_loss = sigmoid_focal_loss_jit(src_logits, target_classes_onehot,
                                              alpha=self.focal_loss_alpha,
                                              gamma=self.focal_loss_gamma,
                                              reduction="none")
            loss_ce = torch.sum(cls_loss) / num_boxes
        else:
            loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="sum") / num_boxes

        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']
        batch_size = len(targets)

        pred_norm_box_list = []
        tgt_box_list = []
        pred_box_list = []
        tgt_box_xyxy_list = []

        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue

            bz_image_whwh = targets[batch_idx]['image_size_xyxy']
            bz_src_boxes = src_boxes[batch_idx]
            bz_target_boxes = targets[batch_idx]["boxes"]
            bz_target_boxes_xyxy = targets[batch_idx]["boxes_xyxy"]

            pred_norm_box_list.append(bz_src_boxes[valid_query] / bz_image_whwh)
            tgt_box_list.append(bz_target_boxes[gt_multi_idx])

            pred_box_list.append(bz_src_boxes[valid_query])
            tgt_box_xyxy_list.append(bz_target_boxes_xyxy[gt_multi_idx])

        if len(pred_norm_box_list) != 0:
            src_boxes_norm = torch.cat(pred_norm_box_list)
            target_boxes = torch.cat(tgt_box_list)

            loss_bbox = F.l1_loss(src_boxes_norm, box_cxcywh_to_xyxy(target_boxes), reduction='none')
            losses = {'loss_bbox': loss_bbox.sum() / num_boxes}
            src_boxes_abs = torch.cat(pred_box_list)
            target_boxes_abs = torch.cat(tgt_box_xyxy_list)
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes_abs, target_boxes_abs))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            losses = {'loss_bbox': outputs['pred_boxes'].sum() * 0,
                      'loss_giou': outputs['pred_boxes'].sum() * 0}

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices, _ = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, _ = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels': kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class HungarianMatcherLight(nn.Module):
    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, use_focal: bool = False):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.ota_k = cfg.matcher.ota_k
        if self.use_focal:
            self.focal_loss_alpha = cfg.matcher.alpha
            self.focal_loss_gamma = cfg.matcher.gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        if self.use_focal:
            out_prob = outputs["pred_logits"].sigmoid()
            out_bbox = outputs["pred_boxes"]
        else:
            out_prob = outputs["pred_logits"].softmax(-1)
            out_bbox = outputs["pred_boxes"]

        indices = []
        matched_ids = []

        for batch_idx in range(bs):
            bz_boxes = out_bbox[batch_idx]
            bz_out_prob = out_prob[batch_idx]
            bz_tgt_ids = targets[batch_idx]["labels"]
            num_insts = len(bz_tgt_ids)

            if num_insts == 0:
                non_valid = torch.zeros(bz_out_prob.shape[0]).to(bz_out_prob) > 0
                indices.append((non_valid, torch.arange(0, 0).to(bz_out_prob)))
                matched_ids.append(torch.arange(0, 0).to(bz_out_prob))
                continue

            bz_gtboxs_abs_xyxy = targets[batch_idx]['boxes_xyxy']

            fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
                box_xyxy_to_cxcywh(bz_boxes),
                box_xyxy_to_cxcywh(bz_gtboxs_abs_xyxy),
                expanded_strides=32
            )

            pair_wise_ious = ops.box_iou(bz_boxes, bz_gtboxs_abs_xyxy)

            # Class Cost
            if self.use_focal:
                alpha = self.focal_loss_alpha
                gamma = self.focal_loss_gamma
                neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
            else:
                cost_class = -bz_out_prob[:, bz_tgt_ids]

            bz_image_size_out = targets[batch_idx]['image_size_xyxy']
            bz_image_size_tgt = targets[batch_idx]['image_size_xyxy_tgt']
            bz_out_bbox_ = bz_boxes / bz_image_size_out
            bz_tgt_bbox_ = bz_gtboxs_abs_xyxy / bz_image_size_tgt
            cost_bbox = torch.cdist(bz_out_bbox_, bz_tgt_bbox_, p=1)

            cost_iou = -pair_wise_ious

            cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_iou + 100.0 * (
                ~is_in_boxes_and_center)
            cost[~fg_mask] = cost[~fg_mask] + 10000.0

            indices_batchi, matched_qidx = self.dynamic_k_matching_light(cost, pair_wise_ious, num_insts)

            indices.append(indices_batchi)
            matched_ids.append(matched_qidx)

        return indices, matched_ids

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        xy_target_gts = box_cxcywh_to_xyxy(target_gts)
        anchor_center_x = boxes[:, 0].unsqueeze(1)
        anchor_center_y = boxes[:, 1].unsqueeze(1)

        b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0)
        b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
        b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)

        is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_boxes_all = is_in_boxes.sum(1) > 0

        center_radius = 2.5
        b_l = anchor_center_x > (
                    target_gts[:, 0] - (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = anchor_center_x < (
                    target_gts[:, 0] + (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = anchor_center_y > (
                    target_gts[:, 1] - (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = anchor_center_y < (
                    target_gts[:, 1] + (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching_light(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = self.ota_k

        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        matched_query_counts = matching_matrix.sum(1)
        if (matched_query_counts > 1).any():
            ambiguous_indices = torch.where(matched_query_counts > 1)[0]
            min_cost_indices = torch.argmin(cost[ambiguous_indices], dim=1)
            matching_matrix[ambiguous_indices] = 0.0
            matching_matrix[ambiguous_indices, min_cost_indices] = 1.0
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        matched_query_id = torch.zeros(num_gt, dtype=torch.long, device=cost.device)

        return (selected_query, gt_indices), matched_query_id