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
         --epochs 7

         --transitive-pairs 10

         --use-kfold
         --folds 5

         --use-images
         --conv-layers 1 4 4 8 16
         --conv-model Bottleneck

         --fc-layers 1024 16
         --dropout       0.7
         --auto-find-lr
         --sc-milestones 3 10
         --weight-decay 0.05

				 --use-distance
				 --d-layers 16 1
				 --d-dropout 0
         --gpus=4
'
mkdir log/${SLURM_JOBID}
cp scripts/script-Image.sh log/${SLURM_JOBID}/
cat scripts/script-Image.sh
echo 'Starting Shell Script with the following options'
source /cluster/home/dzhu/.bashrc
conda activate pytorchbug
python pytorch_src/train_lightning.py $OPTIONS
echo 'Python script finished.'
mv outputs/*image-${SLURM_JOBID}.txt log/${SLURM_JOBID}/
