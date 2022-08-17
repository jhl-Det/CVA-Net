#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

python setup.py build install
# +cu111
# conda install pytorch=1.8.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch