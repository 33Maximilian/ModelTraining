#!/bin/bash
# 真机数据集多卡训练

python -c "import torch; torch.cuda.empty_cache()"

cd lerobot
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=4 \
  $(which lerobot-train) \
  --dataset.repo_id=${YOUR_DATASET_NAME} \
  --dataset.root=${YOUR_DATASET_ROOT} \
  \
  --policy.type=pi05 \
  --policy.device=cuda \
  \
  --peft.target_modules='[q_proj,v_proj]' \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  \
  --policy.compile_model=true \
  --policy.pretrained_path=../pi05_base_model \
  --policy.gradient_checkpointing=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  \
  --steps=100000 \
  --batch_size=4 \
  --policy.use_peft=True \
  --peft.method_type=lora \
  --peft.r=16 \
  \
  --output_dir=../outputs/pi05_real_lora \
  --job_name=pi05_real_lora \
  \
  --wandb.enable=true \
  --wandb.project=pi05_real_lora \
  \
  --policy.repo_id=${YOUR_PROJECT_NAME}/lerobot/pi05_real_lora \
  --policy.push_to_hub=false