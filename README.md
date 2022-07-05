# CVA-Net
## The datasets is available at https://pan.baidu.com/s/1yYME7-DvvIEZzCb72NXaJA?pwd=jnie
The code of this repository is based on
https://github.com/fundamentalvision/Deformable-DETR.

## Installation
### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n cva_net python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate cva_net
    ```
  

* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```


### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
# Deformable DETR

By [Xizhou Zhu](https://scholar.google.com/citations?user=02RXI00AAAAJ),  [Weijie Su](https://www.weijiesu.com/),  [Lewei Lu](https://www.linkedin.com/in/lewei-lu-94015977/), [Bin Li](http://staff.ustc.edu.cn/~binli/), [Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/), [Jifeng Dai](https://jifengdai.org/).

This repository is an official implementation of the paper [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159).


## Introduction

**TL; DR.** Deformable DETR is an efficient and fast-converging end-to-end object detector. It mitigates the high complexity and slow convergence issues of DETR via a novel sampling-based efficient attention mechanism.  

![deformable_detr](./figs/illustration.png)

![deformable_detr](./figs/convergence.png)

**Abstract.** Breast lesion detection in ultrasound is critical for breast cancer
diagnosis. Existing methods mainly rely on individual 2D ultrasound images or
combine unlabeled video and labeled 2D images to train models for breast lesion
detection. In this paper, we first collect and annotate an ultrasound video
dataset (188 videos) for breast lesion detection. Moreover, we propose a
clip-level and video-level feature aggregated network (CVA-Net) for addressing
breast lesion detection in ultrasound videos by aggregating video-level lesion
classification features and clip-level temporal features. The clip-level
temporal features encode local temporal information of ordered video frames and
global temporal information of shuffled video frames. In our CVA-Net, an
inter-video fusion module is devised to fuse local features from original video
frames and global features from shuffled video frames, and an intra-video
fusion module is devised to learn the temporal information among adjacent video
frames. Moreover, we learn video-level features to classify the breast lesions
of the original video as benign or malignant lesions to further enhance the
final breast lesion detection performance in ultrasound videos. Experimental
results on our annotated dataset demonstrate that our CVA-Net clearly
outperforms state-of-the-art methods.

## License

This project is released under the [Apache 2.0 license](./LICENSE).

## Changelog

See [changelog.md](./docs/changelog.md) for detailed logs of major changes. 


## Citing Deformable DETR
If you find Deformable DETR useful in your research, please consider citing:
```bibtex
@article{zhu2020deformable,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
```



*Note:*

1. All models of Deformable DETR are trained with total batch size of 32. 
2. Training and inference speed are measured on NVIDIA Tesla V100 GPU.
3. "Deformable DETR (single scale)" means only using res5 feature map (of stride 32) as input feature maps for Deformable Transformer Encoder.
4. "DC5" means removing the stride in C5 stage of ResNet and add a dilation of 2 instead.
5. "DETR-DC5+" indicates DETR-DC5 with some modifications, including using Focal Loss for bounding box classification and increasing number of object queries to 300.
6. "Batch Infer Speed" refer to inference with batch size = 4  to maximize GPU utilization.
7. The original implementation is based on our internal codebase. There are slight differences in the final accuracy and running time due to the plenty details in platform switch.


## Installation

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## Usage

### Dataset preparation

Please download [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:

```
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

### Training

#### Training on single node

For example, the command for training Deformable DETR on 8 GPUs is as following:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh
```

#### Training on multiple nodes

For example, the command for training Deformable DETR on 2 nodes of each with 8 GPUs is as following:

On node 1:

```bash
MASTER_ADDR=<IP address of node 1> NODE_RANK=0 GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 16 ./configs/r50_deformable_detr.sh
```

On node 2:

```bash
MASTER_ADDR=<IP address of node 1> NODE_RANK=1 GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 16 ./configs/r50_deformable_detr.sh
```

#### Training on slurm cluster

If you are using slurm cluster, you can simply run the following command to train on 1 node with 8 GPUs:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh <partition> deformable_detr 8 configs/r50_deformable_detr.sh
```

Or 2 nodes of  each with 8 GPUs:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh <partition> deformable_detr 16 configs/r50_deformable_detr.sh
```
#### Some tips to speed-up training
* If your file system is slow to read images, you may consider enabling '--cache_mode' option to load whole dataset into memory at the beginning of training.
* You may increase the batch size to maximize the GPU utilization, according to GPU memory of yours, e.g., set '--batch_size 3' or '--batch_size 4'.

### Evaluation

You can get the config file and pretrained model of Deformable DETR (the link is in "Main Results" session), then run following command to evaluate it on COCO 2017 validation set:

```bash
<path to config file> --resume <path to pre-trained model> --eval
```

You can also run distributed evaluation by using ```./tools/run_dist_launch.sh``` or ```./tools/run_dist_slurm.sh```.
