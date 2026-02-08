#!/bin/bash

cd lerobot

lerobot-eval \
  --policy.pretrained_path=../outputs/pi05_libero_lora/checkpoints/last/pretrained_model \
  --policy.type=pi05 \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --output_dir=../outputs/eval/pi05_libero_lora/