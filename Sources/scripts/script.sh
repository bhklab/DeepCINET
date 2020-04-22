#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mem=40G
#SBATCH -o deepnet_bot_fc_1dis.out
#SBATCH -e deepnet_bot_fc_1dis.err
#SBATCH -J DeepCINET_nodist
#SBATCH -c 32
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gpus=1
OPTIONS=''
echo 'Starting Shell Script'
source /cluster/home/dzhu/.bashrc
python pytorch_src/train_lightning.py $OPTIONS
echo 'Python script finished.'
