# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import List
import torch
import torch.nn.functional as F

from detrex.modeling import SetCriterion
from detrex.utils import get_world_size, is_dist_avail_and_initialized
from detrex.layers import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


class DeformableCriterion(SetCriterion):
    """This class computes the loss for Deformable-DETR
    and two-stage Deformable-DETR
    """

    def __init__(
        self,
        num_classes,
        matcher,
        enc_matcher,
        weight_dict,
        losses: List[str] = ["class", "boxes"],
        eos_coef: float = 0.1,
        loss_class_type: str = "focal_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super(DeformableCriterion, self).__init__(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            eos_coef=eos_coef,
            loss_class_type=loss_class_type,
            alpha=alpha,
            gamma=gamma,
        )
        from .one2manyMatcherDAC import Stage2Assigner
        self.topk_matcher = Stage2Assigner(num_queries=900, max_k=6)
        self.enc_matcher = enc_matcher


    @staticmethod
    def indices_merge(num_queries, o2o_indices, o2m_indices):
        bs = len(o2o_indices)
        temp_indices = torch.zeros(bs, num_queries, dtype=torch.int64).cuda() - 1
        new_one2many_indices = []

        for i in range(bs):
            one2many_fg_inds = o2m_indices[i][0].cuda()
            one2many_gt_inds = o2m_indices[i][1].cuda()
            one2one_fg_inds = o2o_indices[i][0].cuda()
            one2one_gt_inds = o2o_indices[i][1].cuda()

            combined = torch.cat((torch.stack((one2many_fg_inds, one2many_gt_inds), dim=1), torch.stack((one2one_fg_inds, one2one_gt_inds), dim=1)))
            unique_pairs = torch.unique(combined, dim=0)
            fg_inds, gt_inds = unique_pairs[:, 0], unique_pairs[:, 1]

            # temp_indices[i][one2one_fg_inds] = one2one_gt_inds
            # temp_indices[i][one2many_fg_inds] = one2many_gt_inds
            # fg_inds = torch.nonzero(temp_indices[i] >= 0).squeeze(1)
            # gt_inds = temp_indices[i][fg_inds]
            new_one2many_indices.append((fg_inds, gt_inds))

        return new_one2many_indices

    def forward(self, outputs, targets):
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "group" in outputs:
            for i, aux_outputs in enumerate(outputs["group"]):
                # o2o_indices = self.matcher(aux_outputs, targets)
                # o2m_indices = self.topk_matcher(aux_outputs, targets)
                # indices = self.indices_merge(900, o2o_indices, o2m_indices)
                indices = self.topk_matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_group_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if "sep" in outputs:
            for i, aux_outputs in enumerate(outputs["sep"]):
                indices = self.topk_matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_sep_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)


        # Compute losses for two-stage deformable-detr
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])

            # NOTE refer to the implementation of MS-DETR
            # NOTE I don't know if it can improve the performance in Mr.DETR but use it
            enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
            indices = self.enc_matcher(enc_outputs, bin_targets)
            enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']

            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)


            # enc o2m outputs loss
            enc_outputs = outputs["enc_outputs_o2m"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])

            # NOTE refer to the implementation of MS-DETR
            # NOTE I don't know if it can improve the performance in Mr.DETR but use it
            enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
            indices = self.enc_matcher(enc_outputs, bin_targets)
            enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']

            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc_o2m": v for k, v in l_dict.items()}
                losses.update(l_dict)


            # enc o2m outputs loss
            # enc_outputs = outputs["enc_outputs_sep"]
            # bin_targets = copy.deepcopy(targets)
            # for bt in bin_targets:
            #     bt["labels"] = torch.zeros_like(bt["labels"])

            # # NOTE refer to the implementation of MS-DETR
            # enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
            # indices = self.enc_matcher(enc_outputs, bin_targets)
            # enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']

            # for loss in self.losses:
            #     l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
            #     l_dict = {k + "_enc_sep": v for k, v in l_dict.items()}
            #     losses.update(l_dict)

        return losses

# # Dense
# # coding=utf-8
# # Copyright 2022 The IDEA Authors. All rights reserved.
#
# import copy
# from typing import List
# import torch
# import torch.nn.functional as F
#
# from detrex.modeling import SetCriterion
# from detrex.utils import get_world_size, is_dist_avail_and_initialized
# from detrex.layers import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
#
#
# class DeformableCriterion(SetCriterion):
#     """This class computes the loss for Deformable-DETR and two-stage Deformable-DETR"""
#
#     def __init__(
#             self,
#             num_classes,
#             matcher,
#             enc_matcher,
#             weight_dict,
#             losses: List[str] = ["class", "boxes"],
#             eos_coef: float = 0.1,
#             loss_class_type: str = "focal_loss",
#             alpha: float = 0.25,
#             gamma: float = 2.0,
#             mal_alpha: float = 0.5,  # [新增] MAL 的超参
#     ):
#         super(DeformableCriterion, self).__init__(
#             num_classes=num_classes,
#             matcher=matcher,
#             weight_dict=weight_dict,
#             losses=losses,
#             eos_coef=eos_coef,
#             loss_class_type=loss_class_type,
#             alpha=alpha,
#             gamma=gamma,
#         )
#         from .one2manyMatcherDAC import Stage2Assigner
#         self.topk_matcher = Stage2Assigner(num_queries=900, max_k=6)
#         self.enc_matcher = enc_matcher
#         self.mal_alpha = mal_alpha  # [新增]
#
#     # ====================================================================================
#     # [新增] 移植自 DEIM 的 MAL Loss (Matching-Aware Loss)
#     # ====================================================================================
#     def loss_labels_mal(self, outputs, targets, indices, num_boxes, **kwargs):
#         """
#         Matching-Aware Loss for Dense O2O
#         """
#         assert 'pred_boxes' in outputs
#         # 获取匹配索引
#         idx = self._get_src_permutation_idx(indices)
#
#         # 计算 IoU 作为定位质量 (Quality)
#         src_boxes = outputs['pred_boxes'][idx]
#         target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
#         ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
#         ious = torch.diag(ious).detach()
#
#         # 准备分类目标
#         src_logits = outputs['pred_logits']
#         target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                     dtype=torch.int64, device=src_logits.device)
#         target_classes[idx] = target_classes_o
#         target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
#
#         # 构造 soft target (IoU * one_hot)
#         target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
#         target_score_o[idx] = ious.to(target_score_o.dtype)
#         target_score = target_score_o.unsqueeze(-1) * target
#
#         pred_score = torch.sigmoid(src_logits).detach()
#
#         # [MAL 核心公式]
#         # 对正样本，目标分数为 IoU^gamma (或者直接 IoU，DEIM代码中有一行被注释的pow，这里按标准VFL逻辑处理)
#         # DEIM 原版实现：target_score = target_score.pow(self.gamma) (如果启用)
#         # 这里我们保持与 VFL 类似的逻辑，确保数值稳定性
#
#         # 计算权重
#         # weight = alpha * pred^gamma * (1-target) + target
#         # 对于正样本 (target=1): weight = 1 (或者 mal_alpha)
#         # 对于负样本 (target=0): weight = alpha * pred^gamma
#
#         # DEIM 实现逻辑:
#         if self.mal_alpha is not None:
#             weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
#         else:
#             weight = pred_score.pow(self.gamma) * (1 - target) + target
#
#         loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
#         loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
#
#         # 返回 loss 字典，key 必须是 'loss_class' 以兼容权重配置
#         return {'loss_class': loss}
#
#     # ====================================================================================
#
#     def forward(self, outputs, targets):
#         outputs_without_aux = {
#             k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"
#         }
#
#         # -----------------------------------------------------------
#         # 1. 主路径 (Main Path)
#         # [策略] 使用 MAL Loss (O2O + Matching Aware)
#         # -----------------------------------------------------------
#         indices = self.matcher(outputs_without_aux, targets)
#
#         num_boxes = sum(len(t["labels"]) for t in targets)
#         num_boxes = torch.as_tensor(
#             [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
#         )
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_boxes)
#         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
#
#         losses = {}
#         for loss in self.losses:
#             # [修改] 如果是分类损失，对于主路径使用 MAL
#             if loss == 'class':
#                 losses.update(self.loss_labels_mal(outputs, targets, indices, num_boxes))
#             else:
#                 losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
#
#         # 辅助损失 (Main Path inter-layers) - 同样跟随主路径使用 MAL
#         if "aux_outputs" in outputs:
#             for i, aux_outputs in enumerate(outputs["aux_outputs"]):
#                 indices = self.matcher(aux_outputs, targets)
#                 for loss in self.losses:
#                     if loss == 'class':
#                         l_dict = self.loss_labels_mal(aux_outputs, targets, indices, num_boxes)
#                     else:
#                         l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
#                     l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
#                     losses.update(l_dict)
#
#         # -----------------------------------------------------------
#         # 2. 辅助路径 2: Group / O2M
#         # [策略] 保持原状 (使用 Focal Loss / loss_labels)
#         # -----------------------------------------------------------
#         if "group" in outputs:
#             for i, aux_outputs in enumerate(outputs["group"]):
#                 indices = self.topk_matcher(aux_outputs, targets)
#                 for loss in self.losses:
#                     # 这里直接调用 self.get_loss，它会路由到默认的 loss_labels (Focal)
#                     l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
#                     l_dict = {k + f"_group_{i}": v for k, v in l_dict.items()}
#                     losses.update(l_dict)
#
#         # -----------------------------------------------------------
#         # 3. 辅助路径 1: Sep / Independent FFN (Dense O2O)
#         # [策略] 使用 MAL Loss
#         # -----------------------------------------------------------
#         if "sep" in outputs:
#             for i, aux_outputs in enumerate(outputs["sep"]):
#                 indices = self.matcher(aux_outputs, targets)  # O2O Matcher
#                 for loss in self.losses:
#                     if loss == 'class':
#                         # [修改] 使用 MAL
#                         l_dict = self.loss_labels_mal(aux_outputs, targets, indices, num_boxes)
#                     else:
#                         l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
#                     l_dict = {k + f"_sep_{i}": v for k, v in l_dict.items()}
#                     losses.update(l_dict)
#
#         # Two-stage encoder losses (保持原状)
#         if "enc_outputs" in outputs:
#             enc_outputs = outputs["enc_outputs"]
#             bin_targets = copy.deepcopy(targets)
#             for bt in bin_targets:
#                 bt["labels"] = torch.zeros_like(bt["labels"])
#
#             enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
#             indices = self.enc_matcher(enc_outputs, bin_targets)
#             enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
#
#             for loss in self.losses:
#                 l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes)
#                 l_dict = {k + "_enc": v for k, v in l_dict.items()}
#                 losses.update(l_dict)
#
#             # enc o2m outputs loss
#             enc_outputs = outputs["enc_outputs_o2m"]
#             # [FIX] 增加判断，防止没有 enc_outputs_o2m 时报错
#             if "enc_outputs_o2m" in outputs:
#                 enc_outputs = outputs["enc_outputs_o2m"]
#                 bin_targets = copy.deepcopy(targets)
#                 for bt in bin_targets:
#                     bt["labels"] = torch.zeros_like(bt["labels"])
#
#                 # 交换 anchors 和 boxes 以适配 enc_matcher
#                 enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
#                 indices = self.enc_matcher(enc_outputs, bin_targets)
#                 enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
#
#                 for loss in self.losses:
#                     # 使用 MAL Loss 或普通 Loss
#                     if loss == 'class':
#                         # 这里 num_boxes 需要传什么？通常 enc loss 用总 box 数即可
#                         l_dict = self.loss_labels_mal(enc_outputs, bin_targets, indices, num_boxes)
#                     else:
#                         l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes)
#
#                     l_dict = {k + "_enc_o2m": v for k, v in l_dict.items()}
#                     losses.update(l_dict)
#
#
#
#             bin_targets = copy.deepcopy(targets)
#             for bt in bin_targets:
#                 bt["labels"] = torch.zeros_like(bt["labels"])
#
#             enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
#             indices = self.enc_matcher(enc_outputs, bin_targets)
#             enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
#
#             for loss in self.losses:
#                 l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes)
#                 l_dict = {k + "_enc_o2m": v for k, v in l_dict.items()}
#                 losses.update(l_dict)
#
#         return losses