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

         --batch-size 128
         --epochs 20

         --transitive-pairs 15

         --use-kfold
         --folds 5

         --use-images
         --conv-layers 1 8 16 32 64 128
         --conv-model Bottleneck

         --fc-layers 1032 256 128 64  1
         --dropout        0.8 0.7 0.6 0
				 --learning-rate 0.01
         --weight-decay 0.001
         --sc-milestones 10

         --gpus=4
'
mkdir log/${SLURM_JOBID}
cp scripts/script-custom.sh log/${SLURM_JOBID}/
cat scripts/script-custom.sh
echo 'Starting Shell Script with the following options'
source /cluster/home/dzhu/.bashrc
conda activate pytorchbug
python pytorch_src/train_lightning.py $OPTIONS
echo 'Python script finished.'
mv outputs/*image-${SLURM_JOBID}.txt log/${SLURM_JOBID}/
