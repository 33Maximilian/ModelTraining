# Model Training

## 总览

- **主要模块**: ACT、Pi0.5、GR00T N1.6
- **支持策略**: ACT (Action Chunking Transformer)、Pi0.5、GR00T N1.6 Policy
- **项目路径**: `ModelTraining`

### Tips：单卡VS多卡

Pi0.5、GR00T这种规模的训练单卡全参训不动，所以主播用了4卡的云端5090服务器+LoRA finetune来跑。小规模的训练（比如主播的ACT Policy示例）可以单卡直接训，就是慢一点。请根据自己的实际算力情况调整咕咕咕

---

## 项目结构

```
ModelTraining/
├── ACT/                       # ACT 策略训练模块
│   ├── readme.md
│   └── scripts/
│       ├── ALOHA/
│       │   ├── train_aloha_insertion.sh
│       │   ├── train_aloha_transfer.sh
│       │   ├── continue.sh
│       │   ├── eval_aloha.sh
│       │   └── readme.md
│       └── PushT/
│           ├── test_pusht.sh
│           ├── train_pusht.sh
│           ├── continue.sh
│           ├── eval_pusht.sh
│           └── readme.md
├── Pi05/                      # Pi05 策略训练模块
│   ├── readme.md
│   └── scripts/
│       ├── train_pi05_sim.sh
│       ├── train_pi05_real.sh
│       └── eval_pi05.sh
├── GR00T_N16/                 # GR00T N1.6 策略训练模块
│   ├── readme.md
│   └── pyproject.toml
└── readme.md
```

---

## QuickStart

### 1. 环境配置

#### 克隆和安装

- 按照lerobot官方链接进行安装：https://huggingface.co/docs/lerobot/installation
- 然后在同一根目录下克隆本仓库：

```bash
git clone https://github.com/33Maximilian/ModelTraining.git
```

- 注意：Pi0.5等政策需求ffmpeg 7.1.1版，所以安装ffmpeg时需要执行：

```bash
conda install ffmpeg=7.1.1 -c conda-forge
```

- 很多模型对依赖版本要求非常严格，并且和lerobot官方不太一样，需要注意安装步骤，参照具体任务readme

#### 登录HuggingFace

```bash
# 用于模型上传到Hub
hf auth login
```

### 2. 开始训练


根据需要选择不同的训练任务：

- **PushT 任务**: 见 [ACT/scripts/PushT/readme.md](ACT/scripts/PushT/readme.md)
- **ALOHA 任务**: 见 [ACT/scripts/ALOHA/readme.md](ACT/scripts/ALOHA/readme.md)
- **Pi0 任务**: 见 [Pi05/readme.md](Pi05/readme.md)
- **GR00T N1.6 任务**: 见 [GR00T_N16/readme.md](GR00T_N16/readme.md)

---

### 3. 实时监控训练

#### 配置 WandB

```bash
# 安装 WandB
pip install wandb

# 登录
wandb login
```

访问 https://wandb.ai 创建账户并获取 API key。

启用条件：

```bash
--wandb.enable=true \
--wandb.project=${YOUR_PROJECT_NAME}\
```
**训练开始后**，查看 WandB 输出中的链接：https://wandb.ai/your-username/your_project_name/runs/run_id

如需本地训练而不上传到 WandB，修改脚本中的参数：

```bash
--wandb.enable=false
```