# 文件: detrex/modeling/backbone/mamba_vision_backbone.py

# CHANGE 1: 使用正确的相对导入来找到 mamba_vision_arch.py
from .mamba_vision_arch import MambaVision, mamba_vision_T, mamba_vision_S, mamba_vision_B, mamba_vision_L

import torch.nn as nn
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec


# MambaVisionBackbone 封装类保持不变，它是正确的
class MambaVisionBackbone(Backbone):
    """
    一个封装了原始 MambaVision 模型的 Backbone 类，
    使其能够输出中间特征图以适配 Detectron2。
    """

    def __init__(self, mamba_model: MambaVision, out_features: list[str]):
        super().__init__()
        self.mamba_model = mamba_model
        self._out_features = out_features
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        current_stride = 4

        # 动态获取 base_dim
        # MambaVision的 `dim` 参数是第一阶段的输出维度
        base_dim = self.mamba_model.levels[0].blocks[0].conv1.in_channels

        for i, level in enumerate(self.mamba_model.levels):
            stage_name = f"stage{i + 1}"
            self._out_feature_strides[stage_name] = current_stride
            self._out_feature_channels[stage_name] = int(base_dim * 2 ** i)
            if level.downsample is not None:
                current_stride *= 2

        del self.mamba_model.norm
        del self.mamba_model.avgpool
        del self.mamba_model.head

    def forward(self, x):
        outputs = {}
        x = self.mamba_model.patch_embed(x)
        for i, level in enumerate(self.mamba_model.levels):
            x = level(x)
            stage_name = f"stage{i + 1}"
            if stage_name in self._out_features:
                outputs[stage_name] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


# -------------------------------------------------------------------------
# CHANGE 2: 重构构建函数，使其只返回 bottom-up 部分，并改名为 build_mamba_vision_backbone
# -------------------------------------------------------------------------
@BACKBONE_REGISTRY.register()
def build_mamba_vision_backbone(
        model_name: str = "mamba_vision_T",
        pretrained: bool = True,
        out_features: list[str] = ["stage2", "stage3", "stage4"]
):
    """
    创建 MambaVision 骨干网络实例 (bottom-up)。
    这个函数现在只构建和返回 MambaVisionBackbone，不再包含 FPN。
    """
    # 使用 globals() 通过字符串名称找到对应的模型创建函数
    model_func = globals()[model_name]
    original_mamba_model = model_func(pretrained=pretrained)

    # 将原始模型用我们的封装类包裹起来
    backbone = MambaVisionBackbone(
        mamba_model=original_mamba_model,
        out_features=out_features
    )
    return backbone