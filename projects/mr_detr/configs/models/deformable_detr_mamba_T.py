import torch.nn as nn

from detrex.modeling.backbone import build_mamba_vision_backbone
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine

from projects.mr_detr_deformable.modeling import (
    DeformableDETR,
    DeformableDetrTransformerEncoder,
    DeformableDetrTransformerDecoder,
    DeformableDetrTransformer,
    DeformableCriterion,
)



model = L(DeformableDETR)(
    backbone=L(build_mamba_vision_backbone)(
        model_name="mamba_vision_T",
        pretrained=True, # 预训练权重在这里加载
        out_features=["stage2", "stage3", "stage4"],
    ),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "stage2": ShapeSpec(channels=320),
            "stage3": ShapeSpec(channels=640),
            "stage4": ShapeSpec(channels=640),
        },
        in_features=["stage2", "stage3", "stage4"],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),
    transformer=L(DeformableDetrTransformer)(
        encoder=L(DeformableDetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            post_norm=False,
            num_feature_levels="${..num_feature_levels}",
        ),
        decoder=L(DeformableDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
        ),
        as_two_stage="${..as_two_stage}",
        num_feature_levels=4,
        two_stage_num_proposals="${..num_queries}",
        mixed_selection=True,
    ),
    embed_dim=256,
    num_classes=80,
    num_queries=300,
    aux_loss=True,
    with_box_refine=True,
    as_two_stage=True,
    criterion=L(DeformableCriterion)(
        num_classes=80,
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        enc_matcher=L(HungarianMatcher)(
            cost_class=0.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
    ),
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    select_box_nums_for_evaluation=300,
    device="cuda",
    mixed_selection=True, # tricks
)


weight_dict_enc={
     "loss_class": 2.0 * 3,
     "loss_bbox": 5.0 * 3,
     "loss_giou": 2.0 * 3
}
weight_dict_enc_o2m={
     "loss_class": 2.0 * 3,
     "loss_bbox": 5.0 * 3,
     "loss_giou": 2.0 * 3
}
weight_dict_o2m={
     "loss_class": 2.0,
     "loss_bbox": 5.0,
     "loss_giou": 2.0
}
weight_dict_sep={
     "loss_class": 2.0,
     "loss_bbox": 5.0,
     "loss_giou": 2.0
}
#weight_dict_enc = model.criterion.weight_dict

if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    aux_weight_dict.update({k + "_enc": v for k, v in weight_dict_enc.items()})
    aux_weight_dict.update({k + "_enc_o2m": v for k, v in weight_dict_enc_o2m.items()})
    aux_weight_dict.update({k + "_enc_sep": v for k, v in weight_dict_enc_o2m.items()})
    for i in range(6):
        aux_weight_dict.update({k + f"_group_{i}": v for k, v in weight_dict_o2m.items()})
    for i in range(6):
        aux_weight_dict.update({k + f"_sep_{i}": v for k, v in weight_dict_sep.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict