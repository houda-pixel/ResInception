# Inception-based Deep Learning Architecture for 3D Point Cloud Completion

This repository contains PyTorch implementation for: Inception-based Deep Learning Architecture for 3D Point Cloud Completion. 

## Pretrained Models

We provide pretrained ResInception models:
| dataset  | url| performance |
| --- | --- |  --- |
| ShapeNet-55 | [[Google Drive](https://drive.google.com/drive/folders/14pLl-NcBZuz8rluwvcMe3uOwrNhJQ_iI?usp=sharing)]   | CD-l2 = 0.95|
| ShapeNet-34 | [[Google Drive](https://drive.google.com/drive/folders/1KhaDTD2XCAFmT7b6QAPtgY-E13JDJ2Ui?usp=sharing)]   | CD-l2 = 1.39|
| PCN |  [[Google Drive](https://drive.google.com/drive/folders/1KhaDTD2XCAFmT7b6QAPtgY-E13JDJ2Ui?usp=sharing)]   | CD-l2 = 0.25|


## Usage

### Requirements

- PyTorch >= 1.7.0
- python >= 3.7
- CUDA >= 9.0
- GCC >= 4.9 
- torchvision
- timm
- open3d
- tensorboardX

```
pip install -r requirements.txt
```

#### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN

*NOTE:* PyTorch >= 1.7 and GCC >= 4.9 are required.

```
# Chamfer Distance
bash install.sh
```
The solution for a common bug in chamfer distance installation can be found in Issue [#6](https://github.com/yuxumin/PoinTr/issues/6)
```
# PointNet++
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```


### Dataset

The details of our new ***ShapeNet-55/34*** datasets and other existing datasets can be found in [DATASET.md](./DATASET.md).

### Evaluation

To evaluate a pre-trained PoinTr model on the Three Dataset with single GPU, run:

```
bash ./scripts/test.sh <GPU_IDS>  \
    --ckpts <path> \
    --config <config> \
    --exp_name <name> \
    [--mode <easy/median/hard>]
```

####  Some examples:
Test the ResIncep-PoinTr pretrained model on the PCN benchmark:
```
bash ./scripts/test.sh 0 \
    --ckpts ./PCN/ResIncep-PoinTr/ckpt-best.pth \
    --config ./cfgs/PCN_models/ResIncep-PoinTr.yaml \
    --exp_name example
```
Test the ResIncep-PoinTr pretrained model on ShapeNet55 benchmark (*easy* mode):
```
bash ./scripts/test.sh 0 \
    --ckpts ./ShapeNet-55/ResIncep-PoinTr/ckpt-best.pth \
    --config ./cfgs/ShapeNet55_models/ResIncep-PoinTr.yaml \
    --mode easy \
    --exp_name example
```

### Training

To train a point cloud completion model from scratch, run:

```
# Use DistributedDataParallel (DDP)
bash ./scripts/dist_train.sh <NUM_GPU> <port> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
# or just use DataParallel (DP)
bash ./scripts/train.sh <GPUIDS> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
```
####  Some examples:
Train a ResIncep-PoinTr model on PCN benchmark with 2 gpus:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/PCN_models/ResIncep-PoinTr.yaml \
    --exp_name example
```
Resume a checkpoint:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/PCN_models/ResIncep-PoinTr.yaml \
    --exp_name example --resume
```

Train a ResIncep-PoinTr model with a single GPU:
```
bash ./scripts/train.sh 0 \
    --config ./cfgs/PCN_models/ResIncep-PoinTr.yaml

````

#### Acknowledgment:
This repository is forked from: https://github.com/yuxumin/PoinTr. We thank the authors for their amazing work.
