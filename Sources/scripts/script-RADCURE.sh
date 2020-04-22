#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --mem=40G
#SBATCH -o outputs/output-deep-%j.txt
#SBATCH -e outputs/error-deep-%j.txt
#SBATCH -J DeepCINET
#SBATCH -c 16
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gres=gpu:3
OPTIONS='--clinical-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/clinical_rad_sort.csv

         --use-radiomics
         --radiomics-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/radiomics_st_sort.csv

         --epochs=5

         --batch-size 1024

         --use-kfold
         --folds=20

         --transitive-pairs 10000

         --fc-layers 1556 16  8   4 1
         --dropout        0.7 0.3 0 0

         --gpus 3
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
