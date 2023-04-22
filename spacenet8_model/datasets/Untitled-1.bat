#!/bin/bash
#SBATCH --job-name=my_job_name
#SBATCH --output=my_job_output_%j.txt
#SBATCH --error=my_job_error_%j.txt
#SBATCH --partition=my_partition_name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:4


module load python scipy-stack
module load gcc/9.3.0  cuda/11.4
module load opencv/4.7.0
module load gdal

source ~/deep/bin/activate

# define variables
TASK=foundation_xdxd_sn5
LOG_DIR=/home/mahzare/projects/def-spiteri/mahzare/spacenet/wdata/logs/train
PRETRAINED_PATH=/work/xdxd_sn5_models/xdxd_sn5_serx50_focal
ARGS= --override_model_dir /home/mahzare/projects/def-spiteri/mahzare/spacenet/work/models Data.train_dir=/home/mahzare/projects/def-spiteri/mahzare/spacenet/data/train

# train each fold
for i in {0..3}
do
    srun --gres=gpu:1 nohup env CUDA_VISIBLE_DEVICES=$i python tools/train_net.py \
    --task $TASK \
    --exp_id 8000$i \
    --fold_id $i \
    --artifact_dir /home/mahzare/projects/def-spiteri/mahzare/spacenet/wdata \
    --pretrained_path $PRETRAINED_PATH/fold$i/fold$i_best.pth \
    --disable_wandb \
    $ARGS \
    > $LOG_DIR/exp_8000$i.txt 2>&1 &
done

wait

# train the last fold
srun --gres=gpu:1 nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task $TASK \
    --exp_id 80004 \
    --fold_id 4 \
    --artifact_dir /home/mahzare/projects/def-spiteri/mahzare/spacenet/wdata \
    --pretrained_path $PRETRAINED_PATH/fold0/fold0_best.pth \
    --disable_wandb \
    $ARGS \
    > $LOG_DIR/exp_80004.txt 2>&1 &
