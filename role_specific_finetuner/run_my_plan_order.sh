#!/bin/bash
#SBATCH -A research
#SBATCH -c 10
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu 2G
#SBATCH --time 4-00:00:00
#SBATCH --output my_plan_order.log
#SBATCH --mail-user shivprasad.sagare@research.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name my_plan_order

source ~/miniconda3/etc/profile.d/conda.sh
conda activate xalign_role

python my_plan_order.py