#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mem=24G
#SBATCH -o outputs/output-trans.txt
#SBATCH -e outputs/error-trans.txt
#SBATCH -J Transitive
#SBATCH -c 16
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gpus=1
OPTIONS='--clinical-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/clinical_rad_sort.csv
         --radiomics-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/Preprocessed/RADCURE/radiomics_st_sort.csv
         --epochs=100
         --sc-milestones 5 15 30 50
         --folds=5
         --transitive-pairs 15
         --fc-layers 1556 8   4   4
         --dropout        0.3 0.3 0
         --use-distance
         --d-layers  4 4  1
         --d-dropout   0  0
'

cat scripts/script-Transitive.sh
echo 'Starting Shell Script with the following script'
source /cluster/home/dzhu/.bashrc
python pytorch_src/train_lightning.py $OPTIONS
mkdir log/${SLURM_JOBID}
echo 'Python script finished.'
mv outputs/*trans.txt log/${SLURM_JOBID}/
