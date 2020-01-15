#!/bin/bash
#SBATCH -t 3-00:00:00
#SBATCH --mem=8G
#SBATCH -o deepnet.out
#SBATCH -e deepnet.err
#SBATCH -J DeepCINET
#SBATCH -c 16
#SBATCH -N 1
#SBATCH --partition=all
OPTIONS=' \
          --model ImageSiamse'
echo 'Starting Shell Script'
source /cluster/home/dzhu/.bashrc
python pytorch_src/train.py --model ImageSiamese
echo 'Python script finished.'
