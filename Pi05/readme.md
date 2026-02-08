# Pi05 Policy 训练

## 项目信息

- **策略**: Pi0.5 Policy
- **模型**: Pi0.5
- **项目路径**: `ModelTraining/Pi05`

---

## 目录结构

```
Pi05/
├── readme.md
├── local_dataset                     #下载到本地的数据集
├── pi05_base_model                   #基线模型
├── paligemma_model                   #基线模型
└── scripts/
    ├── train_pi05_sim.sh
    ├── train_pi05_real.sh
    └── eval_pi05.sh
```

---

## QuickStart

### 1. 训练脚本说明

- **train_pi05_sim.sh**: Pi05 仿真训练脚本
- **train_pi05_real.sh**: Pi05 实机数据集训练脚本
- **eval_pi05.sh**: 仿真效果评估
---

### 2. 环境配置

1） 按照 [LeRobot 安装指南](https://huggingface.co/docs/lerobot/installation) 配置基础环境。

2） 激活环境：

```bash
conda activate lerobot
```

这里注意，由于pi0.5对transformers的要求很严格且和lerobot官方不太一样，所以先把其他依赖/插件都装完最后再安装lerobot[pi]。

#### Tip1：5090 显卡特殊环境要求

**由于5080/5090显卡是新的sm120架构，因此对CUDA和PyTorch版本有要求，需要在完成官方环境配置命令的基础上改一下依赖版本**。

参考：https://github.com/huggingface/lerobot/pull/2477

```bash
pip install torch==2.7.1 torchcodec==0.5 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

**手动修改lerobot/requirements-ubuntu.txt**或者按官方教程**全装完再单独修复冲突依赖**都可以，主要就是注意**pytorch和ffmpeg版本问题**。

另外对于多卡训练，还需要确保所有分布式worker的pytorch版本都正确。

3） 安装 libero 仿真器：

```bash
pip install -e ".[libero]"
```

#### Tip2：LoRA finetune

LeRobot目前是支持LoRA参数高效微调（PEFT）的，强烈建议单卡/低性能卡用户在跑Pi0这种大模型时使用LoRA finetune，这样显存占用能降低到大约原来的10%。

**LoRA 优势：**
- 显著降低 VRAM 占用，适合大模型如 Pi0
- 支持更大 batch size 和更高学习率
- 收敛更快，训练更高效

**安装 LoRA 相关依赖：**

```bash
pip install lerobot[peft]
```

**LoRA核心参数：**

- `--policy.use_peft=True`：启用 LoRA/PEFT
- `--peft.method_type=lora`：指定LoRA方法
- `--peft.r=16`：LoRA rank（8-32均可）
- `--peft.target_modules='[q_proj,v_proj]'`：指定LoRA作用层

启用 LoRA 后，batch size 可适当增大，降低显存压力。

4） 安装 Pi0.5 相关依赖：

```bash
pip install -e ".[pi]"
```

#### Tip3：LoRA首次训练不能启动

官方lerbot/src/lerobot/policies/factory.py代码片段如下：

```bash
elif cfg.pretrained_path and cfg.use_peft:
    # Load a pretrained PEFT model on top of the policy. The pretrained path points to the folder/repo
    # of the adapter and the adapter's config contains the path to the base policy. So we need the
    # adapter config first, then load the correct policy and then apply PEFT.
    from peft import PeftConfig, PeftModel

    logging.info("Loading policy's PEFT adapter.")

    peft_pretrained_path = cfg.pretrained_path
    peft_config = PeftConfig.from_pretrained(peft_pretrained_path)
      ...
```

这段假定永远是“加载一个已有的LoRA adapter”，然后去找adapter_config.json，但实际上做LoRA初次微调是不会有adapter目录的。正确来说LoRA首次微调应该只加载底模参数，初始化LoRA adapter配置，然后让PEFT Trainer或相关逻辑插入LoRA任务，如果恢复训练LoRA adapter才load LoRA adapter config。

因此初次跑需要对该文件做以下修改：

```bash
#在函数头（第19行左右）添加这一行
import os

#把make_policy函数的elif cfg.pretrained_path and cfg.use_peft分支（第494行到第515行）改成以下这样（注意缩进！！！）
elif cfg.pretrained_path and cfg.use_peft:
    from peft import PeftConfig, PeftModel

    peft_pretrained_path = cfg.pretrained_path
    adapter_config_file = os.path.join(str(peft_pretrained_path), "adapter_config.json")

    if os.path.isfile(adapter_config_file):
        # 只在adapter_config.json存在时加载adapter（即恢复/推理现有lora adapter）
        logging.info("Loading policy's PEFT adapter.")
        peft_config = PeftConfig.from_pretrained(peft_pretrained_path)

        kwargs["pretrained_name_or_path"] = peft_config.base_model_name_or_path
        if not kwargs["pretrained_name_or_path"]:
            raise ValueError(
                "No pretrained model name found in adapter config. Can't instantiate the pre-trained policy on which "
                "the adapter was trained."
            )
            
        policy = policy_cls.from_pretrained(**kwargs)
        policy = PeftModel.from_pretrained(policy, peft_pretrained_path, config=peft_config)

    else:
        # 没有adapter_config.json，说明是LoRA首次微调场景，直接用底模初始化
        kwargs["pretrained_name_or_path"] = peft_pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
```

---

### 3. 运行训练

```bash
cd ModelTraining
chmod +x Pi05/*.sh
```

#### Tip4：由于网络不稳定需要使用本地下载的模型/数据集文件

云端跑可能会遇到网络链接不稳定的问题，最稳妥的方式是把数据提前都下载到本地，也可以使用镜像访问。但由于pi05_base模型的 policy_postprocessor.json 文件中设置了默认链接到谷歌的PaliGemma模型，所以只下载pi05_base是不够的，还需把PaliGemma模型也一起下载到本地并修改pi05_base中对应的链接参数，否则pi05_base依旧会尝试链接到原来的网址，从而导致报错无法训练。可以用以下方法解决：

方法1: 直接用镜像

```bash
# 一定要关闭代理，并且在训练过程中都不要打开，然后执行以下命令链接到镜像网站
export HF_ENDPOINT=https://hf-mirror.com
```

方法2: 下载到本地

```bash
# 使用 CLI 下载需要的数据集仓库，我下载的是HuggingFaceVLA/libero，也可以改成指定的
huggingface-cli download HuggingFaceVLA/libero \
    --repo-type dataset \
    --local-dir ./local_dataset

# 下载 libero资产再移到正确缓存路径（对于libero仿真环境需要）
huggingface-cli download lerobot/libero-assets \
    --repo-type dataset \
    --local-dir ./libero-assets
cp -r YOUR_PATH_TO/miniconda3/envs/lerobot/lib/python3.10/site-packages/libero/libero/assets/

# 下载 PI05 基线模型
huggingface-cli download lerobot/pi05_base \
    --repo-type model \
    --local-dir ./pi05_base_model

# 下载PaliGemma 模型到本地
huggingface-cli download google/paligemma-3b-pt-224 \
    --repo-type model \
    --local-dir ./paligemma_model

# 修改 policy_preprocessor.json
python3 << 'EOF'
import json

config_file = "pi05_base_model/policy_preprocessor.json"

with open(config_file, 'r') as f:
    config = json.load(f)

# 找到 tokenizer_processor 步骤并修改 tokenizer_name
for step in config['steps']:
    if step.get('registry_name') == 'tokenizer_processor':
        step['config']['tokenizer_name'] = '../paligemma_model'

# 保存修改后的配置
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

EOF

# 验证修改
echo ""
echo "验证修改结果："
cat pi05_base_model/policy_preprocessor.json | grep -A 7 "tokenizer_processor"
---
```

---

解决以上问题以后应该就可以正式进行模型训练了。

```bash
Pi05/train_pi05_sim.sh
# 或
Pi05/train_pi05_real.sh
```

training正式跑起来以后可以在WandB看到这样的效果：
  ![训练监控效果示例](./WandbPic.png)

---

### 4. 主要训练参数说明

- `--policy.type=pi05`：指定使用 π₀.₅ 策略
- `--dataset.repo_id`：训练数据集
- `--policy.pretrained_path=lerobot/pi05_base`：预训练权重（可选：lerobot/pi05_base, lerobot/pi05_libero）
- `--policy.device=cuda`：使用 GPU
- `--policy.compile_model=true`：加速训练
- `--policy.gradient_checkpointing=true`：显存优化
- `--policy.dtype=bfloat16`：混合精度训练
- `--policy.freeze_vision_encoder`：视觉编码器是否冻结
- `--policy.train_expert_only`：是否仅训练 expert 层
- `--steps`：训练步数
- `--batch_size`：批量大小
- `--output_dir`：输出目录
- `--wandb.enable=true`：启用 WandB 监控
- `--wandb.project=...`：WandB 项目名
- `--policy.repo_id=...`：模型上传目标仓库
- `--policy.push_to_hub=false`：是否推送模型到 HuggingFace Hub

更多参数详见 [官方文档](https://huggingface.co/docs/lerobot/pi05)。

---

### 5. 训练监控与模型上传

- **WandB 监控**：
  - 注册并登录 [WandB](https://wandb.ai)
  - 训练脚本自动上传日志

- **模型上传**：
  - 若需上传模型到 HuggingFace Hub，设置 `--policy.push_to_hub=true` 并配置 `--policy.repo_id`

---

## 参考资料
- [LeRobot π₀.₅ (Pi05) 官方文档](https://huggingface.co/docs/lerobot/pi05)
- [OpenPI 项目](https://github.com/Physical-Intelligence/openpi)
- [LeRobot 安装指南](https://huggingface.co/docs/lerobot/installation)