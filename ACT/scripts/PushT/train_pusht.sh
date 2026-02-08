#!/bin/bash

cd lerobot

lerobot-train \
  --dataset.repo_id=lerobot/pusht \
  \
  --env.type=pusht \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=150 \
  --policy.n_action_steps=100 \
  --policy.kl_weight=10 \
  \
  --steps=150000 \
  --batch_size=32 \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_lr_backbone=1e-4 \
  --eval_freq=2500 \
  --eval.n_episodes=50 \
  \
  --save_checkpoint=true \
  --save_freq=5000 \
  \
  --output_dir=../outputs/act_pusht/train/ \
  --job_name=act_pusht \
  \
  --policy.repo_id=${YOUR_PROJECT_NAME}/act_pusht \
  --policy.push_to_hub=false \
  \
  --wandb.enable=true \
  --wandb.project=lerobot_pusht \
