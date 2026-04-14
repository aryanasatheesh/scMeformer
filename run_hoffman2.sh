#!/bin/bash
# Command used to run scMeformer single_cell_regression on Hoffman2
# Note: requires a GPU node. Request one with:
# qrsh -l h_rt=4:00:00,h_data=16G,gpu,V100

module load cuda/11.8
conda activate /u/scratch/a/aryasath/conda/envs/scmeformer

# Current status: hitting PyTorch 2.1 inplace tensor operation error (unresolved)
# The fp16 flag triggers a gradient computation error in METH_CNN.forward()
# Running without --fp16 hits a separate F.relu inplace issue
# Both are being investigated

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 main.py transformer single_cell_regression \
    --exp_name single_cell_regression \
    --learning_rate 0.000176 \
    --batch_size 128 \
    --data_dir ./datasets/ \
    --output_dir ./outputs/demo_model \
    --warmup_steps 10000 \
    --gradient_accumulation_steps 1 \
    --fp16 --local_rank 0 \
    --model_config_file ./config/config.json
