import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from detrex.config import get_config
from .models.deformable_detr_mamba_T import model
from detectron2.data.datasets import register_coco_instances

register_coco_instances(
    name="my_coco_2017_train",
    metadata={},
    json_file="/home/hp/coco/annotations/instances_train2017.json",
    image_root="/home/hp/coco/train2017/"
)

register_coco_instances(
    name="my_coco_2017_val",
    metadata={},
    json_file="/home/hp/coco/annotations/instances_val2017.json",
    image_root="/home/hp/coco/val2017/"
)

dataloader = get_config("common/data/coco_detr.py").dataloader
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train


base_weight_dict = {
    "loss_class": 2.0,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
}

aux_weight_dict = {}
for i in range(6):
    aux_weight_dict.update({
        f"loss_class_{i}": 2.0,
        f"loss_bbox_{i}": 5.0,
        f"loss_giou_{i}": 2.0,
    })

enc_weight_dict = {
    "loss_class_enc": 2.0,
    "loss_bbox_enc": 5.0,
    "loss_giou_enc": 2.0,
    "loss_class_enc_o2m": 2.0,
    "loss_bbox_enc_o2m": 5.0,
    "loss_giou_enc_o2m": 2.0,
}

weight_dict = {**base_weight_dict, **aux_weight_dict, **enc_weight_dict}
model.criterion.weight_dict = weight_dict

train.output_dir = "./output/deformable_detr_SS2D-Dense 12轮_mamba_T_12ep2"

train.max_iter = 360000

lr_multiplier.scheduler.milestones = [240000, 330000]
lr_multiplier.scheduler.values = [1.0, 0.1, 0.01]
lr_multiplier.scheduler.num_updates = 360000

train.eval_period = 30000
train.log_period = 1000
train.checkpointer.period = 30000

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# devices
train.device = "cuda"
model.device = train.device

optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

dataloader.train.dataset.names = "my_coco_2017_train"
dataloader.test.dataset.names = "my_coco_2017_val"
dataloader.evaluator.dataset_name = "my_coco_2017_val"
dataloader.train.num_workers = 8
dataloader.train.total_batch_size = 4
dataloader.evaluator.output_dir = train.output_dir

from detectron2.engine import HookBase
from detectron2.engine import DefaultTrainer
from detectron2.config import instantiate
import logging


class MosaicFadeOutHook(HookBase):

    def __init__(self, close_iter):
        self.close_iter = close_iter
        self.mosaic_off = False
        self.logger = logging.getLogger("detectron2")

    def after_step(self):
        if self.trainer.iter >= self.close_iter and not self.mosaic_off:
            self.logger.info(
                f"🚀 Iteration {self.trainer.iter} reached. Auto-disabling Mosaic Augmentation for finetuning...")
            dataloader.train.mapper.enable_mosaic = False
            dataloader.train.mapper.mosaic_prob = 0.0

            self.logger.info("🔄 Re-building Data Loader with Mosaic OFF...")
            try:
                new_loader = instantiate(dataloader.train)
                self.trainer.train_loader = new_loader
                self.trainer._data_loader_iter = iter(new_loader)
                self.mosaic_off = True
                self.logger.info("✅ Mosaic Augmentation successfully disabled! Training continues with real images.")
            except Exception as e:
                self.logger.error(f"❌ Failed to swap data loader: {e}")
                raise e


class AutoMosaicTrainer(DefaultTrainer):
    def build_hooks(self):
        hooks = super().build_hooks()

        fade_out_iter = 300000
        hooks.append(MosaicFadeOutHook(close_iter=fade_out_iter))
        return hooks

train._target_ = AutoMosaicTrainer

