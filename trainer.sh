# pip3 install pillow==6.1
# conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia


GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/config.sh