# Omni-MambaDet

## Introduction
Omni-MambaDet is an efficient end-to-end object detection framework designed to achieve an optimal Pareto trade-off among inference speed, accuracy, and computational cost. It elegantly bridges global context and dense supervision by combining an Alternating Hybrid Mamba/Transformer topology (ODSS module) with an Asymmetric Dense Supervision (ADS) mechanism.

## Codebase Structure
Our implementation is built upon `detrex` and `detectron2`. The repository is organized as follows:
* `configs/`: Base configuration files for datasets, optimization, and training schedules.
* `detrex/`: The core modeling library, containing our customized MambaVision backbone, ODSS layers.
* `projects/`: Specific model definitions and configuration overrides for Omni-MambaDet.
* `tools/`: Main executable scripts, including `train_net.py` for training and evaluation.
* `demo/`: Inference scripts for evaluating and visualizing model predictions.
* `tests/`: Unit tests for core network components.

## Datasets
Omni-MambaDet is comprehensively evaluated on the following standard benchmarks. 
* **MS COCO 2017**: [https://cocodataset.org/#download](https://cocodataset.org/#download)
* **PASCAL VOC 2007/2012**: [http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/)
