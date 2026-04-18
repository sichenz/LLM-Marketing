#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=llm_marketing_finetune
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=finetune_%j.out
#SBATCH --error=finetune_%j.err
#SBATCH --gres=gpu:a100:1

module purge

singularity exec --nv \
    --overlay /scratch/$USER/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate llm
        cd /scratch/$USER/LLM-Marketing/deepseek_smol
        python3 finetune.py
    "
