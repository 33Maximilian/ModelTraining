#!/bin/bash

cd lerobot

lerobot-train \
  --dataset.repo_id=lerobot/pusht \
  \
  --env.type=pusht \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=100 \
  --policy.n_action_steps=50 \
  --policy.kl_weight=10 \
  \
  --steps=3000 \
  --batch_size=32 \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_lr_backbone=1e-4 \
  --eval_freq=1000 \
  \
  --output_dir=../outputs/act_pusht_test/train/ \
  --job_name=act_pusht_test \
  \
  --policy.repo_id=${YOUR_PROJECT_NAME}/act_pusht_test \
  --policy.push_to_hub=false \
