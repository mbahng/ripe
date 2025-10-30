#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80gb
#SBATCH --time=48:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --output=logs/train_multi_%j.out

modality_reg_coeff=${1:-0}
last_layer_l1_coeff=${2:-0}
margin_loss_coeff=${3:-0.5}
modality_weight_optimization_epochs=${4:-8}
joint_optimization_epochs=${5:-16}

conda activate env
python -m multimodal.train_multi_model \
  --img_backbone_path /usr/project/xtmp/cjb131/cs474/experiments/test/artifacts/hf8ael2x/runs/checkpoints/strong-whale-xmdbkusau/strong-whale-xmdbkusau_best.pth \
  --gen_backbone_path genetic_protopnet_no_weights.pth \
  --dataset_id smol \
  --modality_reg_coeff="$modality_reg_coeff" \
  --last_layer_l1_coeff="$last_layer_l1_coeff" \
  --margin_loss_coeff=$margin_loss_coeff \
  --modality_weight_optimization_epochs="$modality_weight_optimization_epochs" \
  --joint_optimization_epochs="$joint_optimization_epochs" \
  --margin_loss_type="mean" \
  --samples_per_epoch 2048 \
  --batch_size 128 \
  --model_type protopnet

