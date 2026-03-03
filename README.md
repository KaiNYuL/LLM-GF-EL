# GRPO-TX 可训练项目

本项目按照“模块解耦、流程耦合”的原则实现：
- 模块解耦：`data / reward / training / eval / export / serving / agent_adapter` 各层可独立维护；
- 流程耦合：训练主流程固定为 `QLoRA -> GRPO`，GRPO 依赖 QLoRA 产物。

## 目录结构

```text
src/
  config/
  data/
  reward/
  training/
  pipeline/
  eval/
  export/
  serving/
  agent_adapter/
scripts/
configs/
data/examples/
```

## 快速开始

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行全流程（示例）：

```bash
python scripts/run_pipeline.py --config configs/base.yaml --train data/examples/train_sample.jsonl --pref data/examples/preference_words.json
```

3. 单独运行 QLoRA：

```bash
python scripts/train_qlora.py --config configs/qlora.yaml --train data/examples/train_sample.jsonl
```

4. 单独运行 GRPO：

```bash
python scripts/train_grpo.py --config configs/grpo.yaml --train data/examples/train_sample.jsonl --qlora_ckpt outputs/qlora/checkpoint-final --pref data/examples/preference_words.json
```

5. 运行离线评估示例：

```bash
python scripts/eval.py
```

6. 导出 merge + GGUF（真实流程）：

```bash
python scripts/export.py --checkpoint outputs/grpo/checkpoint-final --base_model Qwen/Qwen2-7B-Instruct --out outputs/export --dtype float16 --quant_type Q4_K_M --llama_cpp_dir D:/tools/llama.cpp --model_name grpo_tx
```

## 训练实现说明

- `src/training/qlora_trainer.py`：已接入 `transformers + peft + trl(SFTTrainer)`，可直接执行 QLoRA 训练；
- `src/training/grpo_trainer.py`：已接入 `trl(GRPOTrainer)`，并基于 QLoRA adapter 继续做偏好对齐；
- `src/reward/style_reward.py`：默认奖励为“正负词奖惩 + 相关性约束 + 长度约束”，可按业务继续扩展。

## 运行前提

- 需要可用 GPU 与 CUDA 环境（QLoRA/GRPO 训练场景）；
- 首次运行前建议执行：

```bash
pip install -r requirements.txt
```

- 如果使用 4bit 量化，需确保 `bitsandbytes` 在当前系统可用。
- 如果要导出 GGUF，需准备可用的 `llama.cpp`，并确保存在：
  - `convert_hf_to_gguf.py`
  - `llama-quantize(.exe)` 或 `quantize(.exe)`
