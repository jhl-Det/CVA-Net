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
