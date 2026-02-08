#!/bin/bash

cd lerobot

lerobot-eval \
  --policy.path=../outputs/act_aloha_transfer/train/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=50 \
  --output_dir=../outputs/eval/act_aloha_transfer
