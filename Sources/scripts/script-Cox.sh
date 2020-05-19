#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mem=30G
#SBATCH -o outputs/output-cox-%j.txt
#SBATCH -e outputs/error-cox-%j.txt
#SBATCH -J Volume
#SBATCH -c 16
#SBATCH -N 1
##SBATCH --partition=himem
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gres=gpu:1
OPTIONS='--clinical-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/clinical_rad_sort.csv
         --radiomics-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/radiomics_st_sort.csv
         --use-cox
         --use-folds
         --folds 20
         --use-radiomics
         --mrmr 100
'
mkdir log/${SLURM_JOBID}
cp scripts/script-Cox.sh log/${SLURM_JOBID}/
cat scripts/script-Cox.sh
echo 'Starting Shell Script with the following options'
source /cluster/home/dzhu/.bashrc
conda activate pytorchbug
python train.py $OPTIONS
echo 'Python script finished.'
mv outputs/*cox-${SLURM_JOBID}.txt log/${SLURM_JOBID}/
