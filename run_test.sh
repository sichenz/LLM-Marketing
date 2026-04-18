#!/bin/bash
set -e

echo "======================================"
echo "Testing SMALL model training pipeline..."
echo "======================================"
torchrun --nproc_per_node=1 --master_port 29501 train.py --model_size small --batch_size 1 --max_steps 2

echo ""
echo "======================================"
echo "Testing LARGE model training pipeline..."
echo "======================================"
torchrun --nproc_per_node=1 --master_port 29502 train.py --model_size large --batch_size 1 --max_steps 2

echo "Both tests completed successfully."
