import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from detrex.config import get_config
from .models.deformable_detr_r50 import model
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

train.init_checkpoint = "/home/hp/jmm/Mr.DETR/Mr.DETR-main/MrDETR_deformable_r50_12ep_300q.pth"
train.output_dir = "./output/deformable_detr_r50_12ep2"

train.max_iter = 720000

lr_multiplier.scheduler.milestones = [480000, 640000]

train.eval_period = 5000

train.log_period = 50

train.checkpointer.period = 30000

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

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


dataloader.train.total_batch_size = 2

dataloader.evaluator.output_dir = train.output_dir
