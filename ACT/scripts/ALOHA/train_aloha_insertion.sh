#!/bin/bash

cd lerobot

lerobot-train \
  --dataset.image_transforms.enable=true \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=100 \
  --policy.n_action_steps=50 \
  \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  \
  --output_dir=../outputs/act_aloha_insertion/train \
  --job_name=act_aloha_insertion \
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
  --wandb.project=aloha_insertion \
  \
  --policy.repo_id=${YOUR_PROJECT_NAME}/act_aloha_insertion \
  --policy.push_to_hub=true \