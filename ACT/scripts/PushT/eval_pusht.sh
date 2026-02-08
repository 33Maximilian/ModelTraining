#!/bin/bash

cd lerobot

lerobot-eval \
  --policy.path=../outputs/act_pusht/train/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --env.type=pusht \
  --env.task=PushT-v0 \
  --eval.n_episodes=50 \
  --output_dir=../outputs/eval/act_pusht \
  --seed=1000
