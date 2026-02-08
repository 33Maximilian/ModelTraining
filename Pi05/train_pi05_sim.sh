#!/bin/bash
# 添加了LoRA finetune来减少显存占用

python -c "import torch; torch.cuda.empty_cache()" # 清缓存防爆

cd lerobot
export MUJOCO_GL=egl  # 对于云端服务器/headless模式
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # PyTorch分段分配模式
export TOKENIZERS_PARALLELISM=false  # 禁用 huggingface 的 tokenizers fork 警告

# 仿真训练需指定env类型和任务
torchrun --nproc_per_node=4 \
  $(which lerobot-train) \
  --dataset.repo_id=HuggingFaceVLA/libero \
  \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  \
  --policy.type=pi05 \
  --policy.device=cuda \
  \
  --policy.use_peft=True \
  --peft.method_type=lora \
  --peft.r=16 \
  --peft.target_modules='[q_proj,v_proj]' \
  \
  --policy.compile_model=true \
  --policy.pretrained_path=../pi05_base_model \
  --policy.gradient_checkpointing=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  \
  --steps=50000 \
  --batch_size=4 \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  \
  --output_dir=../outputs/pi05_libero_lora \
  --job_name=pi05_libero_lora \
  \
  --wandb.enable=true \
  --wandb.project=pi05_libero_lora \
  \
  --policy.repo_id=${YOUR_PROJECT_NAME}/lerobot/pi05_libero_lora \
  --policy.push_to_hub=false