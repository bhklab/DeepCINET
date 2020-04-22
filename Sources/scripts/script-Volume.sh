#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mem=40G
#SBATCH -o outputs/output-volume.txt
#SBATCH -e outputs/error-volume.txt
#SBATCH -J Volume
#SBATCH -c 16
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gpus=1
OPTIONS='--clinical-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/clinical_rad_sort.csv
         --radiomics-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/radiomics_st_sort.csv
         --use-cox
         --use-clinical
'

echo 'Starting Shell Script with the following options'
echo $OPTIONS
source /cluster/home/dzhu/.bashrc
python pytorch_src/train_lightning.py $OPTIONS
echo 'Python script finished.'
