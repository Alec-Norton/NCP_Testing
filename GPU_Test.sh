#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=5g
#SBATCH -J "CfC_NCP_Testing - Alec Norton"
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:2
#SBATCH -C A100|V100

module load python/3.10.2
python3 -m venv myenv
source myenv/bin/activate
nvidia-smi
pip install --upgrade pip
pip install tensorflow[and-cuda]

module load cuda12.2

python GPU_Test.py