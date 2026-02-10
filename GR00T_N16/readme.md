# GR00T N1.6 Policy 训练

## 项目信息

- **策略**: GR00T N1.6 Policy
- **模型**: GR00T N1.6
- **项目路径**: `ModelTraining/Isaac-GR00T`

---

## QuickStart

### 1. 环境配置

1） 按照 [GR00T 安装指南](https://github.com/NVIDIA/Isaac-GR00T/tree/main) 完成仓库克隆，安装uv并激活环境：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --python 3.10
uv pip install -e .
```

2）安装lerobot（用于处理数据集格式）

```bash
sudo apt update
sudo apt install ffmpeg

git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv pip install -e .
uv pip install 'lerobot[all]'
```

---

#### Tip1：5090 显卡特殊环境要求

**由于5080/5090显卡是新的sm120架构，因此对CUDA和PyTorch版本有要求，需要在完成官方环境配置命令的基础上改一下依赖版本，将官方的Isaac-GR00T/pyproject.toml替换为我提供的[pyproject.toml](../GR00T_N16/pyproject.toml)**。

参考：https://github.com/NVIDIA/Isaac-GR00T/issues/523

---

3）安装 GR00T N1.6 相关依赖：

```bash
uv pip install flash-attn==2.8.0.post2
uv pip install transformers==4.51.3
```

4）登陆huggingface和WandB

```bash
uv pip install "huggingface_hub[cli]"
source .venv/bin/activate
huggingface-cli login
deactivate

uv pip install wandb
uv run wandb login
```

---

#### Tip4：由于网络不稳定需要使用本地下载的模型/数据集文件

云端跑可能会遇到网络链接不稳定的问题，最稳妥的方式是把数据提前都下载到本地，也可以使用镜像访问。可以用以下方法解决：

方法1: 直接用镜像

```bash
# 一定要关闭代理，并且在训练过程中都不要打开，然后执行以下命令链接到镜像网站
export HF_ENDPOINT=https://hf-mirror.com
```

方法2: 下载到本地

```bash
huggingface-cli download nvidia/GR00T-N1.6-3B --local-dir ${YOUR_OUTPUT_PATH}
```

解决以上问题以后应该就可以正式进行模型训练了。

---

### 2. 运行训练

接下来按照官方文档一步一步复现即可，请根据自己设备的实际情况调整官方脚本中对应参数。

1）仿真微调实例：[Fine-tune LIBERO](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/LIBERO/README.md#finetune-libero-spatial-dataset)

2）实机数据集微调：[Fine-tune on Custom Embodiments](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md)

---

### 3. 主要训练参数说明

参考 [Fine-tune LIBERO](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/LIBERO/README.md#finetune-libero-spatial-dataset)。

#### （1）运行环境参数

- `set -x -e`  
  Bash命令，输出每条执行命令并在遇到错误时立即退出。
- `export NUM_GPUS=${N}`  
- `--num_gpus $NUM_GPUS`  
- `torchrun ... --nproc_per_node=$NUM_GPUS --master_port=29500`  
  设置使用n张GPU，用于启动PyTorch分布式训练启动器，--nproc_per_node 指定每节点进程数，--master_port 设置分布式主端口。

#### （2）模型与数据

- `--base_model_path ${YOUR_MODEL_PATH}`  
  预训练基线模型的本地目录/路径（支持 Hugging Face Hub id 或绝对/相对目录），用于finetune或加载权重。
- `--dataset_path ${YOUR_DATASET_PATH}`  
  指定数据集的本地路径，用于加载训练/评估数据。
- `--embodiment_tag LIBERO_PANDA`  
  指定使用的机器人/任务形态（embodiment），如用于不同机械臂/任务组的特殊处理。

#### （3）结果与模型存储

- `--output_dir ${YOUR_OUTPUT_PATH}`  
  训练结果保存目录，包括模型权重、配置、日志、检查点等。
- `--save_steps 1000`  
  每间隔1000步保存一次模型checkpoint。
- `--save_total_limit 5`  
  最多只保存5个模型checkpoint文件，旧的会被覆盖/删除。

#### （4）训练流程控制

- `--max_steps 20000`  
  训练最大步数，对应优化迭代轮数/数据样本的总数量。
- `--warmup_ratio 0.05`  
  学习率warmup策略，前5%训练步数线性增加学习率。
- `--weight_decay 1e-5`  
  权重衰减参数（L2正则化），防止模型过拟合。
- `--learning_rate 1e-4`  
  优化器的初始学习率。

#### （5）监控与批量设置

- `--use_wandb`  
  启用 Weights & Biases 云端实验监控（需账户和API KEY）。
- `--global_batch_size ${N}`  
  所有进程合成的总批次大小。

#### （6）数据与增强

- `--color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08`  
  图像增强参数，随机亮度、对比度、饱和度、色调调整范围。
- `--dataloader_num_workers 4`  
  数据加载进程数，提升数据预处理与加载速度。

#### （7）模型正则与Dropout

- `--state_dropout_prob 0.8`  
  模型state部分的dropout概率，提升泛化能力，减少过拟合。

---

## 参考资料
- [Isaac-GR00T 官方文档](https://github.com/NVIDIA/Isaac-GR00T/tree/main)
- [Fine-tune LIBERO](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/LIBERO/README.md#finetune-libero-spatial-dataset)
- [Finetune New Embodiment](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md)