from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RuntimeConfig:
    """运行时配置。"""

    seed: int = 42
    output_root: str = "outputs"
    log_with: str = "tensorboard"


@dataclass
class QLoRAConfig:
    """QLoRA 训练配置。"""

    model_name_or_path: str = "Qwen/Qwen2-7B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_seq_length: int = 1024
    gradient_checkpointing: bool = True
    logging_steps: int = 10


@dataclass
class GRPOConfig:
    """GRPO 训练配置。"""

    num_generations: int = 4
    learning_rate: float = 2e-5
    beta: float = 0.04
    kl_coeff: float = 0.05
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_completion_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    num_train_epochs: int = 1
    logging_steps: int = 10


@dataclass
class AppSettings:
    """应用配置总入口。"""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置字典。"""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_settings(config_path: str | Path | None = None) -> AppSettings:
    """加载配置。

    规则：
    1. 先使用 dataclass 默认值；
    2. 再叠加 YAML 文件中的值（若提供）；
    3. 最后应用少量环境变量覆盖。
    """
    data: dict[str, Any] = {
        "runtime": RuntimeConfig().__dict__,
        "qlora": QLoRAConfig().__dict__,
        "grpo": GRPOConfig().__dict__,
    }

    if config_path:
        cfg_file = Path(config_path)
        if cfg_file.exists():
            with cfg_file.open("r", encoding="utf-8") as fp:
                yaml_data = yaml.safe_load(fp) or {}
            data = _deep_update(data, yaml_data)

    if os.getenv("OUTPUT_ROOT"):
        data["runtime"]["output_root"] = os.environ["OUTPUT_ROOT"]

    return AppSettings(
        runtime=RuntimeConfig(**data["runtime"]),
        qlora=QLoRAConfig(**data["qlora"]),
        grpo=GRPOConfig(**data["grpo"]),
    )
