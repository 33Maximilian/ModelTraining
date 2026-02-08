# ALOHA ACT 训练

## 项目信息

- **任务**: ALOHA 双臂机械臂操作任务
- **策略**: ACT
- **数据集**: 
  - lerobot/aloha_sim_insertion_human (插入任务)
  - lerobot/aloha_sim_transfer_cube_human (传递任务)
- **项目路径**: `ModelTraining/ACT/scripts/ALOHA`

---

## QuickStart

### 1. 使用训练脚本

- **train_aloha_insertion.sh**: 插入任务训练
- **train_aloha_transfer.sh**: 传递任务训练
- **continue.sh**: 断点续训
- **eval_aloha.sh**: 模型评估


### 2. 运行训练

```bash
conda activate lerobot
cd ModelTraining

# 赋予执行权限
chmod +x ACT/scripts/ALOHA/*.sh

# 训练插入任务
ACT/scripts/ALOHA/train_aloha_insertion.sh

# 训练传递任务
ACT/scripts/ALOHA/train_aloha_transfer.sh

# 断点续训
ACT/scripts/ALOHA/continue.sh
```

---

## 评估模型

```bash
# 使用评估脚本（默认评估传递任务）
ACT/scripts/ALOHA/eval_aloha.sh

# 评估插入任务
python ACT/scripts/lerobot_eval.py \
  --policy.path=ACT/outputs/act_aloha_insertion/train/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --eval.n_episodes=50 \
  --eval.batch_size=50 \
  --output_dir=ACT/outputs/eval/act_aloha_insertion
```

---

## 关键参数说明

### train_*.sh 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--dataset.image_transforms.enable` | true | 启用图像变换 |
| `--dataset.repo_id` | lerobot/* | 数据集仓库ID |
| `--policy.type` | act | 策略类型 |
| `--policy.device` | cuda | 运行设备 |
| `--policy.chunk_size` | 100 | 动作chunk大小 |
| `--policy.n_action_steps` | 50 | 单个动作步数 |
| `--policy.kl_weight` | 10 | KL散度权重 |
| `--policy.hidden_dim` | 512 | 隐层维度 |
| `--env.type` | aloha | 环境类型 |
| `--env.task` | * | 环境任务 |
| `--output_dir` | ../outputs/*/train | 输出目录 |
| `--job_name` | * | 任务名称 |
| `--steps` | 150000 | 训练总步数 |
| `--policy.optimizer_lr` | 1e-4 | 优化器学习率 |
| `--policy.optimizer_lr_backbone` | 1e-4 | backbone学习率 |
| `--batch_size` | 32 | 批次大小 |
| `--eval_freq` | 2500 | 评估频率（步数） |
| `--save_checkpoint` | true | 是否保存checkpoint |
| `--save_freq` | 5000 | 保存频率（步数） |
| `--wandb.enable` | true | 启用WandB |
| `--wandb.project` | * | WandB项目名称 |
| `--policy.repo_id` | ${YOUR_PROJECT_NAME}/* | HuggingFace仓库ID |
| `--policy.push_to_hub` | true | 是否上传到Hub |


### continue.sh 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--config_path` | ../outputs/*/train/checkpoints/last/pretrained_model/train_config.json | 训练配置文件 |
| `--resume` | true | 启用断点续训 |
| `--steps` | 150000 | 目标训练步数 |
| `--policy.optimizer_lr` | 1e-4 | 优化器学习率 |
| `--policy.optimizer_lr_backbone` | 1e-4 | backbone学习率 |
| `--wandb.enable` | false | 禁用WandB |

---

## 任务说明

### AlohaInsertion-v0 (插入任务)
- **数据集**: lerobot/aloha_sim_insertion_human
- **目标**: 双臂协作完成物体插入任务
- **输出目录**: ModelTraining/outputs/act_aloha_insertion/train/
- **Hub仓库**: ${YOUR_PROJECT_NAME}/act_aloha_insertion
- **训练步数**: 150000
- **评估频率**: 每2500步
- **保存频率**: 每5000步

### AlohaTransferCube-v0 (传递任务)
- **数据集**: lerobot/aloha_sim_transfer_cube_human
- **目标**: 双臂协作传递方块
- **输出目录**: ModelTraining/outputs/act_aloha_transfer/train/
- **Hub仓库**: ${YOUR_PROJECT_NAME}/act_aloha_transfer
- **训练步数**: 150000
- **评估频率**: 每2500步
- **保存频率**: 每5000步

---

## 相关文档

- LeRobot GitHub: https://github.com/huggingface/lerobot
- ACT文档: https://github.com/huggingface/lerobot/blob/main/docs/source/act.mdx
- ALOHA数据集: 
  - https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human
  - https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human
- PushT数据集:
  - https://huggingface.co/datasets/lerobot/pusht
