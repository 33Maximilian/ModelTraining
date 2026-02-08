#!/bin/bash

cd lerobot

lerobot-train \
  --dataset.image_transforms.enable=true \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=100 \
  --policy.n_action_steps=50 \
  \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  \
  --output_dir=../outputs/act_aloha_transfer/train \
  --job_name=act_aloha_transfer \
  \
  --steps=150000 \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_lr_backbone=1e-4 \
  --policy.kl_weight=10 \
  --batch_size=32 \
  --eval_freq=2500 \
  \
  --save_checkpoint=true \
  --save_freq=5000 \
  \
  --wandb.enable=true \
  --wandb.project=aloha_transfer_cube \
  \
  --policy.repo_id=${YOUR_PROJECT_NAME}/act_aloha_transfer \
  --policy.push_to_hub=true \
