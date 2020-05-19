#!/bin/bash
#SBATCH -t 71:59:59
#SBATCH --mem=16G
#SBATCH -o outputs/output-deep-%j.txt
#SBATCH -e outputs/error-deep-%j.txt
#SBATCH -J DeepCINET
#SBATCH -c 16
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gres=gpu:1
OPTIONS='
         --clinical-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/clinical_rad_sort.csv

         --use-radiomic
	       --mrmr 30
         --radiomics-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/radiomics_st_sort.csv

         --min-epochs 20
         --max-epochs 20
         --transitive-pairs 10000

         --batch-size 256
         --use-kfold
         --folds 20
         --auto-find-lr

         --fc-layers 30 128  64  64  32 16  1
         --dropout      0.4  0.2 0.2  0  0  0
         --weight-decay 0
         --sc-milestones 10
         --check-val-every-n-epoch 1
         --gpus 1
'
mkdir log/${SLURM_JOBID}
cp scripts/script-RADCURE.sh log/${SLURM_JOBID}/
cat scripts/script-RADCURE.sh
echo 'Starting Shell Script with the following options'
source /cluster/home/dzhu/.bashrc
conda activate pytorchbug
python train.py $OPTIONS
echo 'Python script finished.'
mv outputs/*deep-${SLURM_JOBID}.txt log/${SLURM_JOBID}/
