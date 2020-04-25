#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --mem=16G
#SBATCH -o outputs/output-deep-%j.txt
#SBATCH -e outputs/error-deep-%j.txt
#SBATCH -J DeepCINET
#SBATCH -c 16
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gres=gpu:1
OPTIONS='--clinical-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/clinical_rad_sort.csv

         --use-clinical
         --use-radiomics
         --radiomics-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/radiomics_st_sort.csv

         --epochs 100
         --transitive-pairs 10000

         --batch-size 1024
         --use-kfold
         --folds 5
         --learning-rate 0.01

				 --mrmr 10
         --fc-layers 18 64 64 64 16 1
         --dropout      0  0  0  0  0
         --weight-decay 0
         --sc-milestones 1000
				 --check-val-every-n-epoch 10
         --gpus 1
'
mkdir log/${SLURM_JOBID}
cp scripts/script-RADCURE.sh log/${SLURM_JOBID}/
cat scripts/script-RADCURE.sh
echo 'Starting Shell Script with the following options'
source /cluster/home/dzhu/.bashrc
conda activate pytorchbug
python pytorch_src/train_lightning.py $OPTIONS
echo 'Python script finished.'
mv outputs/*deep-${SLURM_JOBID}.txt log/${SLURM_JOBID}/
