# PushT ACT 训练

## 项目信息

- **任务**: PushT 2D推方块任务
- **策略**: ACT
- **数据集**: lerobot/pusht (官方公开数据集)
- **项目路径**: `ModelTraining/ACT/scripts/PushT`

---

## QuickStart

### 1. 使用训练脚本

- **test_pusht.sh**: 快速测试
- **train_pusht.sh**: 完整训练
- **continue.sh**: 断点续训
- **eval_pusht.sh**: 模型评估


### 2. 运行训练

```bash
conda activate lerobot
cd ModelTraining

# 赋予执行权限
chmod +x ACT/scripts/PushT/*.sh

# 快速测试
ACT/scripts/PushT/test_pusht.sh

# 完整训练
ACT/scripts/PushT/train_pusht.sh

# 断点续训
ACT/scripts/PushT/continue.sh
```

---

## 评估模型

```bash
# 使用评估脚本
ACT/scripts/PushT/eval_pusht.sh
```

---

## 关键参数说明（对标脚本代码）

### train_*.sh 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--dataset.repo_id` | lerobot/pusht | 数据集仓库ID |
| `--env.type` | pusht | 环境类型 |
| `--policy.type` | act | 策略类型 |
| `--policy.device` | cuda | 运行设备 |
| `--policy.chunk_size` | 100 | 动作chunk大小 |
| `--policy.n_action_steps` | 50 | 单个动作步数 |
| `--policy.kl_weight` | 10 | KL散度权重 |
| `--policy.hidden_dim` | 512 | 隐层维度 |
| `--steps` | 100000 | 训练总步数 |
| `--batch_size` | 32 | 批次大小 |
| `--policy.optimizer_lr` | 1e-4 | 优化器学习率 |
| `--policy.optimizer_lr_backbone` | 1e-4 | backbone学习率 |
| `--eval_freq` | 2500 | 评估频率（步数） |
| `--save_checkpoint` | true | 是否保存checkpoint |
| `--save_freq` | 5000 | 保存频率（步数） |
| `--output_dir` | ../outputs/act_pusht/train/ | 输出目录 |
| `--job_name` | act_pusht | 任务名称 |
| `--policy.repo_id` | ${YOUR_PROJECT_NAME}/act_pusht | HuggingFace仓库ID |
| `--policy.push_to_hub` | false | 是否上传到Hub |
| `--wandb.enable` | true | 启用WandB |
| `--wandb.project` | lerobot_pusht | WandB项目名称 |

### continue.sh 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--config_path` | ../outputs/act_pusht/train/checkpoints/last/pretrained_model/train_config.json | 训练配置文件 |
| `--resume` | true | 启用断点续训 |
| `--steps` | 150000 | 目标训练步数 |
| `--policy.optimizer_lr` | 1e-4 | 优化器学习率 |
| `--policy.optimizer_lr_backbone` | 1e-4 | backbone学习率 |
| `--wandb.enable` | false | 禁用WandB |

---

## 说明

- **test_pusht.sh**: 快速测试模式，10000步验证环境和代码
- **train_pusht.sh**: 完整训练，输出存放在 `act_pusht` 目录
- **continue.sh**: 从上一次训练的last checkpoint继续，最终训练至150000步
- 断点续训会自动加载之前的模型权重和优化器状态

---

## 相关文档

- LeRobot GitHub: https://github.com/huggingface/lerobot
- ACT文档: https://github.com/huggingface/lerobot/blob/main/docs/source/act.mdx
- PushT数据集: https://huggingface.co/datasets/lerobot/pusht
- PushT GitHub: https://github.com/huggingface/gym-pusht
