#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH --mem=160G
#SBATCH -o outputs/output-image-%j.txt
#SBATCH -e outputs/error-image-%j.txt
#SBATCH -J DeepCINET-Image
#SBATCH -c 32
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gres=gpu:4
OPTIONS='--clinical-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/RADCURE/clinical.csv
         --radiomics-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/RAadiomics_HN2/radiomics.csv
         --image-path=/cluster/home/dzhu/Documents/DATA/UHN-Project/Radiomics_HN2/RADCURE/images/

         --batch-size 8
         --num-workers 32
         --min-epochs 5
         --max-epochs 5

         --transitive-pairs 30
         --use-volume-cache
         --accumulate-grad-batches 10

         --folds 3

         --use-images
         --conv-layers 1 8  8  8  16 16 32 32
         --pool          1  1  1  1  1  1  1
         --conv-model ResNet

         --fc-layers 128 64  1
         --dropout       0.3 0
         --auto-find-lr
         --weight-decay 0.001
         --sc-milestones 10
         --check-val-every-n-epoch 1
         --gpus=4
'
mkdir log/${SLURM_JOBID}
cp scripts/script-Image.sh log/${SLURM_JOBID}/
cat scripts/script-Image.sh
echo 'Starting Shell Script with the following options'
source /cluster/home/dzhu/.bashrc
conda activate pytorchbug
python train.py $OPTIONS
echo 'Python script finished.'
mv outputs/*image-${SLURM_JOBID}.txt log/${SLURM_JOBID}/
