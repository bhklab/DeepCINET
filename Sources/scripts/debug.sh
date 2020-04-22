#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mem=2G
#SBATCH -o debug.out
#SBATCH -e debug.err
#SBATCH -J DeepCINETDist
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --partition=all
##SBATCH --account=radiomics_gpu
##SBATCH --partition=gpu_radiomics
##SBATCH --gpus=1
OPTIONS='--clinical-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/clinical_rad_sort_debug.csv \
         --radiomics-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/radiomics_st_sort_debug.csv \
         --epochs=1 \
         --fc-layers 1556 16  32  64  32  32  1 \
         --dropout        0   0.9 0.9 0.7 0.7 0'
echo 'Starting Shell Script'
source /cluster/home/dzhu/.bashrc
python pytorch_src/train_lightning.py $OPTIONS
echo 'Python script finished.'
