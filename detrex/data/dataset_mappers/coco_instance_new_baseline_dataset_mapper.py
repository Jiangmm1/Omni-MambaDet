# # coding=utf-8
# # Copyright 2022 The IDEA Authors. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ------------------------------------------------------------------------------------------------
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# # ------------------------------------------------------------------------------------------------
# # COCO Instance Segmentation with LSJ Augmentation
# # Modified from:
# # https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py
# # ------------------------------------------------------------------------------------------------
#
# import copy
# import logging
# import numpy as np
# import torch
#
# from detectron2.structures import Instances, Boxes, PolygonMasks
# from detectron2.data import detection_utils as utils
# from detectron2.data import transforms as T
#
# from pycocotools import mask as coco_mask
#
#
# def convert_coco_poly_to_mask(segmentations, height, width):
#     masks = []
#     for polygons in segmentations:
#         rles = coco_mask.frPyObjects(polygons, height, width)
#         mask = coco_mask.decode(rles)
#         if len(mask.shape) < 3:
#             mask = mask[..., None]
#         mask = torch.as_tensor(mask, dtype=torch.uint8)
#         mask = mask.any(dim=2)
#         masks.append(mask)
#     if masks:
#         masks = torch.stack(masks, dim=0)
#     else:
#         masks = torch.zeros((0, height, width), dtype=torch.uint8)
#     return masks
#
#
# def build_transform_gen(
#     image_size,
#     min_scale,
#     max_scale,
#     random_flip: str = "horizontal",
#     is_train: bool = True,
# ):
#     """
#     Create a list of default :class:`Augmentation`.
#     Now it includes resizing and flipping.
#
#     Returns:
#         list[Augmentation]
#     """
#     assert is_train, "Only support training augmentation."
#     assert random_flip in ["none", "horizontal", "vertical"], f"Only support none/horizontal/vertical flip, but got {random_flip}"
#
#     augmentation = []
#
#     if random_flip != "none":
#         augmentation.append(
#             T.RandomFlip(
#                 horizontal=random_flip == "horizontal",
#                 vertical=random_flip == "vertical",
#             )
#         )
#
#     augmentation.extend([
#         T.ResizeScale(
#             min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size,
#         ),
#         T.FixedSizeCrop(crop_size=(image_size, image_size))
#     ])
#
#     return augmentation
#
#
# class COCOInstanceNewBaselineDatasetMapper:
#     """
#     A callable which takes a dataset dict in Detectron2 Dataset format,
#     and map it into a format used by MaskFormer.
#
#     This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.
#
#     The callable currently does the following:
#
#     1. Read the image from "file_name"
#     2. Applies geometric transforms to the image and annotation
#     3. Find and applies suitable cropping to the image and annotation
#     4. Prepare image and annotation to Tensors
#     """
#     def __init__(
#         self,
#         is_train=True,
#         *,
#         augmentation,
#         image_format,
#     ):
#         self.augmentation = augmentation
#         logging.getLogger(__name__).info(
#             "[COCO_Instance_LSJ_Augment_Dataset_Mapper] Full TransformGens used in training: {}".format(str(self.augmentation))
#         )
#
#         self.img_format = image_format
#         self.is_train = is_train
#
#     def __call__(self, dataset_dict):
#         """
#         Args:
#             dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
#
#         Returns:
#             dict: a format that builtin models in detectron2 accept
#         """
#         dataset_dict = copy.deepcopy(dataset_dict)
#         image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
#         utils.check_image_size(dataset_dict, image)
#
#         padding_mask = np.ones(image.shape[:2])
#         image, transforms = T.apply_transform_gens(self.augmentation, image)
#
#         padding_mask = transforms.apply_segmentation(padding_mask)
#         padding_mask = ~ padding_mask.astype(bool)
#
#         image_shape = image.shape[:2]
#
#         # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
#         # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
#         # Therefore it's important to use torch.Tensor.
#         dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
#         dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
#
#         if not self.is_train:
#             # USER: Modify this if you want to keep them for some reason.
#             dataset_dict.pop("annotations", None)
#             return dataset_dict
#
#         if "annotations" in dataset_dict:
#             for anno in dataset_dict["annotations"]:
#                 anno.pop("keypoints", None)
#
#             annos = [
#                 utils.transform_instance_annotations(obj, transforms, image_shape)
#                 for obj in dataset_dict.pop("annotations")
#                 if obj.get("iscrowd", 0) == 0
#             ]
#             # NOTE: does not support BitMask due to augmentation
#             # Current BitMask cannot handle empty objects
#             instances = utils.annotations_to_instances(annos, image_shape)
#             # After transforms such as cropping are applied, the bounding box may no longer
#             # tightly bound the object. As an example, imagine a triangle object
#             # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
#             # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
#             # the intersection of original bounding box and the cropping box.
#             if not instances.has('gt_masks'):
#                 instances.gt_masks = PolygonMasks([])  # for negative examples
#             instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
#             # Need to filter empty instances first (due to augmentation)
#             instances = utils.filter_empty_instances(instances)
#             # Generate masks from polygon
#             h, w = instances.image_size
#             # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
#             if hasattr(instances, 'gt_masks'):
#                 gt_masks = instances.gt_masks
#                 gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
#                 instances.gt_masks = gt_masks
#             # import ipdb; ipdb.set_trace()
#             dataset_dict["instances"] = instances
#
#         return dataset_dict



# Dense
# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
# ------------------------------------------------------------------------------------------------
# COCO Instance Segmentation with LSJ Augmentation & Mosaic (Dense O2O)
# ------------------------------------------------------------------------------------------------

import copy
import logging
import numpy as np
import torch
import random
from PIL import Image

from detectron2.structures import Instances, Boxes, PolygonMasks
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from pycocotools import mask as coco_mask
import torchvision.transforms.v2 as Tv2
import torchvision.transforms.v2.functional as Tv2F
# 导入 tv_tensors 用于正确的坐标变换
from torchvision import tv_tensors


# =================================================================================
# 1. Mosaic 增强类 (修复坐标缩放问题)
# =================================================================================
class Mosaic(Tv2.Transform):
    def __init__(self, output_size=640, max_size=None, rotation_range=0, translation_range=(0.1, 0.1),
                 scaling_range=(0.5, 1.5), probability=1.0, fill_value=114) -> None:
        super().__init__()
        self.resize = Tv2.Resize(size=output_size, max_size=max_size)
        self.probability = probability
        self.affine_transform = Tv2.RandomAffine(degrees=rotation_range, translate=translation_range,
                                                 scale=scaling_range, fill=fill_value)

    def _get_size_func(self, img):
        if hasattr(Tv2F, "get_size"):
            return Tv2F.get_size(img)
        return Tv2F.get_spatial_size(img)

    def load_samples_from_dataset(self, image, target, dataset_wrapper):
        # image 和 target 进来时已经是 wrapper 过的了，直接 resize
        # Tv2.Resize 会自动处理 image 和 target['boxes'] 的同步缩放
        image, target = self.resize(image, target)

        resized_images, resized_targets = [image], [target]
        max_height, max_width = self._get_size_func(resized_images[0])

        if dataset_wrapper.dataset is None:
            raise ValueError("Dataset is not initialized in the Mapper.")

        indices = range(len(dataset_wrapper.dataset))
        sample_indices = random.choices(indices, k=3)

        for idx in sample_indices:
            # load_item 现在返回的是已经 wrap 好的 boxes
            s_img, s_target = dataset_wrapper.load_item(idx)

            # [CRITICAL FIX] 因为 s_target['boxes'] 已经是 TVTensors，这里 resize 会正确缩放坐标
            s_img, s_target = self.resize(s_img, s_target)

            height, width = self._get_size_func(s_img)
            max_height, max_width = max(max_height, height), max(max_width, width)
            resized_images.append(s_img)
            resized_targets.append(s_target)

        return resized_images, resized_targets, max_height, max_width

    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        mode = images[0].mode if hasattr(images[0], 'mode') else 'RGB'
        merged_image = Image.new(mode=mode, size=(max_width * 2, max_height * 2), color=0)

        for i, img in enumerate(images):
            merged_image.paste(img, placement_offsets[i])

        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)
        merged_target = {}

        if len(targets) > 0:
            keys = targets[0].keys()
            for key in keys:
                if key == 'boxes':
                    # targets[i]['boxes'] 是 TVTensor，加 offset 需要注意
                    values = []
                    for i, target in enumerate(targets):
                        # 加上偏移量 (x, y)
                        box = target[key] + offsets[i]
                        values.append(box)

                    if len(values) > 0:
                        merged_target[key] = torch.cat(values, dim=0)
                else:
                    values = [target[key] for target in targets]
                    if len(values) > 0 and isinstance(values[0], torch.Tensor):
                        merged_target[key] = torch.cat(values, dim=0)
                    else:
                        merged_target[key] = values

        return merged_image, merged_target

    def forward(self, image, target, dataset_wrapper):
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target

        resized_images, resized_targets, max_height, max_width = self.load_samples_from_dataset(image, target,
                                                                                                dataset_wrapper)
        mosaic_image, mosaic_target = self.create_mosaic_from_dataset(resized_images, resized_targets, max_height,
                                                                      max_width)

        # 此时 mosaic_target['boxes'] 已经是正确的坐标了，直接做仿射变换
        # 重新 wrap 一下以防万一 (Concat 后可能变成了纯 Tensor)
        if 'boxes' in mosaic_target and not isinstance(mosaic_target['boxes'], tv_tensors.BoundingBoxes):
            mosaic_target['boxes'] = tv_tensors.BoundingBoxes(
                mosaic_target['boxes'],
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=mosaic_image.size[::-1]
            )

        mosaic_image, mosaic_target = self.affine_transform(mosaic_image, mosaic_target)
        return mosaic_image, mosaic_target


# =================================================================================

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(
        image_size,
        min_scale,
        max_scale,
        random_flip: str = "horizontal",
        is_train: bool = True,
):
    assert is_train, "Only support training augmentation."
    assert random_flip in ["none", "horizontal",
                           "vertical"], f"Only support none/horizontal/vertical flip, but got {random_flip}"

    augmentation = []

    if random_flip != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=random_flip == "horizontal",
                vertical=random_flip == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size,
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size))
    ])

    return augmentation


class COCOInstanceNewBaselineDatasetMapper:
    def __init__(
            self,
            is_train=True,
            *,
            augmentation,
            image_format,
            dataset=None,
            enable_mosaic=False,
            mosaic_prob=1.0,
            output_size=640
    ):
        self.augmentation = augmentation
        logging.getLogger(__name__).info(
            "[COCO_Instance_LSJ_Augment_Dataset_Mapper] Full TransformGens used in training: {}".format(
                str(self.augmentation))
        )

        self.img_format = image_format
        self.is_train = is_train

        self.enable_mosaic = enable_mosaic and is_train
        self.dataset = dataset
        if self.enable_mosaic:
            logging.getLogger(__name__).info(f"Mosaic Augmentation Enabled with prob {mosaic_prob}")
            self.mosaic_transform = Mosaic(
                output_size=output_size,
                probability=mosaic_prob
            )

    def load_item(self, idx):
        if self.dataset is None:
            raise ValueError("Dataset not set in Mapper")
        dataset_dict = copy.deepcopy(self.dataset[idx])
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        image = Image.fromarray(image)

        target = {}
        if "annotations" in dataset_dict:
            annos = dataset_dict.pop("annotations")
            annos = [obj for obj in annos if obj.get("iscrowd", 0) == 0]

            if len(annos) > 0:
                boxes = [obj["bbox"] for obj in annos]
                boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                # XYWH -> XYXY
                boxes[:, 2:] += boxes[:, :2]

                # [CRITICAL FIX] 必须在这里把 Box 包装成 TVTensors，否则 Resize 不会缩放 Box 坐标！
                target['boxes'] = tv_tensors.BoundingBoxes(
                    boxes,
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=image.size[::-1]  # (H, W)
                )

                classes = [obj["category_id"] for obj in annos]
                target['labels'] = torch.as_tensor(classes, dtype=torch.int64)
            else:
                # 即使为空也要包装，保持类型一致
                target['boxes'] = tv_tensors.BoundingBoxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=image.size[::-1]
                )
                target['labels'] = torch.zeros((0,), dtype=torch.int64)

        return image, target

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # ========================================================================
        # Mosaic 增强 (Dense Input)
        # ========================================================================
        if self.is_train and self.enable_mosaic and self.dataset is not None:
            pil_image = Image.fromarray(image)
            target_dict = {}
            has_annos = "annotations" in dataset_dict

            if has_annos:
                annos = [obj for obj in dataset_dict["annotations"] if obj.get("iscrowd", 0) == 0]
                if len(annos) > 0:
                    boxes = [obj["bbox"] for obj in annos]
                    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                    boxes[:, 2:] += boxes[:, :2]  # XYWH -> XYXY

                    # [CRITICAL FIX] 主图的 Box 也必须包装
                    target_dict['boxes'] = tv_tensors.BoundingBoxes(
                        boxes,
                        format=tv_tensors.BoundingBoxFormat.XYXY,
                        canvas_size=pil_image.size[::-1]
                    )
                    classes = [obj["category_id"] for obj in annos]
                    target_dict['labels'] = torch.as_tensor(classes, dtype=torch.int64)
                else:
                    target_dict['boxes'] = tv_tensors.BoundingBoxes(
                        torch.zeros((0, 4), dtype=torch.float32),
                        format=tv_tensors.BoundingBoxFormat.XYXY,
                        canvas_size=pil_image.size[::-1]
                    )
                    target_dict['labels'] = torch.zeros((0,), dtype=torch.int64)

            # 调用 Mosaic
            mosaic_image, mosaic_target = self.mosaic_transform(pil_image, target_dict, self)

            image = np.array(mosaic_image)

            # 解析回 Detrex/Detectron2 格式
            if 'boxes' in mosaic_target and len(mosaic_target['boxes']) > 0:
                boxes_xyxy = mosaic_target['boxes']
                classes = mosaic_target['labels']
                new_annos = []
                for i in range(len(boxes_xyxy)):
                    box = boxes_xyxy[i].tolist()
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    new_annos.append({
                        "bbox": [box[0], box[1], w, h],
                        "bbox_mode": 1,  # BoxMode.XYWH_ABS
                        "category_id": classes[i].item(),
                        "iscrowd": 0
                    })
                dataset_dict["annotations"] = new_annos
            else:
                dataset_dict["annotations"] = []

            dataset_dict["height"], dataset_dict["width"] = image.shape[:2]
        # ========================================================================

        padding_mask = np.ones(image.shape[:2])
        image, transforms = T.apply_transform_gens(self.augmentation, image)

        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            instances = utils.annotations_to_instances(annos, image_shape)

            # [FIXED] 移除了导致崩坏的 Mask 自动生成代码
            # if not instances.has('gt_masks'): ... (deleted)

            instances = utils.filter_empty_instances(instances)

            # 只有当 dataset 中确实包含分割信息时才处理 masks
            # 对于纯检测任务，instances.has('gt_masks') 应该为 False
            h, w = instances.image_size
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks

            dataset_dict["instances"] = instances

        return dataset_dict