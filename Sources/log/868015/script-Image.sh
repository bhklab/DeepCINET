#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --mem=80G
#SBATCH -o outputs/output-image-%j.txt
#SBATCH -e outputs/error-image-%j.txt
#SBATCH -J DeepCINET-Image
#SBATCH -c 32
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gres=gpu:4
OPTIONS='--clinical-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/clinical_rad_images_sort.csv
         --radiomics-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/radiomics_st_images_sort.csv
         --image-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/RADCURE-64/

         --batch-size 64
         --epochs 7

         --transitive-pairs 50

         --use-kfold
         --folds 5

         --use-images
         --conv-layers 1 4 4 4 4

         --fc-layers 256 8   4   1
         --dropout       0.6 0.3 0

         --gpus=4
'
mkdir log/${SLURM_JOBID}
cp scripts/script-Image.sh log/${SLURM_JOBID}/
cat scripts/script-Dist.sh
echo 'Starting Shell Script with the following options'
source /cluster/home/dzhu/.bashrc
python pytorch_src/train_lightning.py $OPTIONS
echo 'Python script finished.'
mv outputs/*image-${SLURM_JOBID}.txt log/${SLURM_JOBID}/
