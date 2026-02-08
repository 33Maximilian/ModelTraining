#!/bin/bash

cd lerobot

python src/lerobot/scripts/lerobot_train.py \
  --config_path=../outputs/act_pusht/train/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --steps=150000 \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_lr_backbone=1e-4 \
  --wandb.enable=false