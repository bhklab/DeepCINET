#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mem=40G
#SBATCH -o outputs/output-deep.txt
#SBATCH -e outputs/error-deep.txt
#SBATCH -J DeepCINET
#SBATCH -c 16
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gpus=1
OPTIONS='--clinical-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/clinical_rad_sort.csv
         --radiomics-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/radiomics_st_sort.csv
         --epochs=7
         --folds=10
         --transitive-pairs 10000
         --fc-layers 1564 16  8   4 1
         --dropout        0.7 0.3 0 0
         --use-clinical
'
cat scripts/script-RADCURE.sh
echo 'Starting Shell Script with the following options'
source /cluster/home/dzhu/.bashrc
python pytorch_src/train_lightning.py $OPTIONS
echo 'Python script finished.'
mkdir log/${SLURM_JOBID}
mv outputs/*deep.txt log/${SLURM_JOBID}/
cp scripts/script-RADCURE.sh log/${SLURM_JOBID}/

