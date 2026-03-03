from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.config.settings import QLoRAConfig


@dataclass
class QLoRATrainConfig:
    """QLoRA 训练参数。

    这里复用 settings 中的字段结构，方便通过配置文件统一管理。
    """

    model_name_or_path: str
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: str
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: int
    max_seq_length: int
    gradient_checkpointing: bool
    logging_steps: int

    @classmethod
    def from_settings(cls, cfg: QLoRAConfig) -> "QLoRATrainConfig":
        return cls(**cfg.__dict__)


class QLoRATrainer:
    """QLoRA 训练器。"""

    def __init__(self, config: QLoRATrainConfig, output_root: str) -> None:
        self.config = config
        self.output_root = Path(output_root)

    def train(self, train_path: str, eval_path: str | None = None) -> str:
        """执行 QLoRA 训练并返回 checkpoint 路径。"""
        train_file = Path(train_path)
        if not train_file.exists():
            raise FileNotFoundError(f"训练数据不存在: {train_path}")

        eval_file = Path(eval_path) if eval_path else None
        if eval_file and not eval_file.exists():
            raise FileNotFoundError(f"评估数据不存在: {eval_path}")

        dtype = _resolve_dtype(self.config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        model.config.use_cache = False
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        train_dataset = _load_sft_dataset(str(train_file))
        eval_dataset = _load_sft_dataset(str(eval_file)) if eval_file else None

        stage_output = self.output_root / "qlora"
        stage_output.mkdir(parents=True, exist_ok=True)
        train_args = SFTConfig(
            output_dir=str(stage_output),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            max_seq_length=self.config.max_seq_length,
            logging_steps=self.config.logging_steps,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset is not None else "no",
            bf16=self.config.bnb_4bit_compute_dtype.lower() == "bfloat16",
            fp16=self.config.bnb_4bit_compute_dtype.lower() == "float16",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            tokenizer=tokenizer,
            dataset_text_field="text",
        )
        trainer.train()

        ckpt_dir = self.output_root / "qlora" / "checkpoint-final"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))

        metadata = {
            "stage": "qlora",
            "train_path": str(train_file),
            "eval_path": eval_path,
            "config": self.config.__dict__,
            "base_model": self.config.model_name_or_path,
            "artifact": "LoRA adapter checkpoint",
        }
        with (ckpt_dir / "meta.json").open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False, indent=2)

        return str(ckpt_dir)


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _build_training_text(item: dict) -> str:
    """将标准样本拼接为 SFT 文本。"""
    system_prompt = (item.get("system_prompt") or "").strip()
    prompt = (item.get("prompt") or "").strip()
    reference = (item.get("reference") or "").strip()

    sections: list[str] = []
    if system_prompt:
        sections.append(f"<|system|>\n{system_prompt}")
    sections.append(f"<|user|>\n{prompt}")
    if reference:
        sections.append(f"<|assistant|>\n{reference}")
    else:
        sections.append("<|assistant|>\n")
    return "\n".join(sections)


def _load_sft_dataset(path: str):
    dataset = load_dataset("json", data_files=path, split="train")
    return dataset.map(lambda row: {"text": _build_training_text(row)})
