#! /bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:turing:1
#SBATCH -w worker2
#SBATCH -e /home/rusek/ltc/log/%x.%a.err
#SBATCH -o /home/rusek/ltc/log/%x.%a.out

cd /home/rusek/ltc