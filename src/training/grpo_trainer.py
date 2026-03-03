from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from trl import GRPOConfig as TRLGRPOConfig
from trl import GRPOTrainer as TRLGRPOTrainer

from src.config.settings import GRPOConfig
from src.data.schemas import PreferenceWords
from src.reward.base import RewardEngine


@dataclass
class GRPOTrainConfig:
    """GRPO 训练参数。"""

    num_generations: int
    learning_rate: float
    beta: float
    kl_coeff: float
    batch_size: int
    gradient_accumulation_steps: int
    max_completion_length: int
    temperature: float
    top_p: float
    num_train_epochs: int
    logging_steps: int

    @classmethod
    def from_settings(cls, cfg: GRPOConfig) -> "GRPOTrainConfig":
        return cls(**cfg.__dict__)


class GRPOTrainer:
    """GRPO 训练器。

    关键点：
    - 输入必须是 QLoRA 阶段产物（checkpoint）；
    - 奖励计算通过 RewardEngine 注入；
    - 训练产物独立落盘，便于回滚与追踪。
    """

    def __init__(self, config: GRPOTrainConfig, output_root: str) -> None:
        self.config = config
        self.output_root = Path(output_root)

    def train(
        self,
        train_path: str,
        qlora_checkpoint_dir: str,
        reward_engine: RewardEngine,
        preference_words: PreferenceWords,
        eval_path: str | None = None,
    ) -> str:
        """执行 GRPO 训练并返回 checkpoint 路径。"""
        train_file = Path(train_path)
        eval_file = Path(eval_path) if eval_path else None
        qlora_ckpt = Path(qlora_checkpoint_dir)
        if not train_file.exists():
            raise FileNotFoundError(f"训练数据不存在: {train_path}")
        if eval_file and not eval_file.exists():
            raise FileNotFoundError(f"评估数据不存在: {eval_path}")
        if not qlora_ckpt.exists():
            raise FileNotFoundError(f"QLoRA checkpoint 不存在: {qlora_checkpoint_dir}")

        model = AutoPeftModelForCausalLM.from_pretrained(
            str(qlora_ckpt),
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(qlora_ckpt), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_dataset = load_dataset("json", data_files=str(train_file), split="train")
        if "prompt" not in train_dataset.column_names:
            raise ValueError("GRPO 训练数据必须包含 prompt 字段")

        eval_dataset = load_dataset("json", data_files=str(eval_file), split="train") if eval_file else None

        stage_output = self.output_root / "grpo"
        stage_output.mkdir(parents=True, exist_ok=True)
        training_args = _build_grpo_config(self.config, str(stage_output))

        def reward_func(prompts, completions, **kwargs):
            prompt_texts = [_normalize_prompt_text(item) for item in prompts]
            completion_texts = [_normalize_completion_text(item) for item in completions]
            references = kwargs.get("reference")
            return reward_engine.score(
                prompts=prompt_texts,
                completions=completion_texts,
                positive_words=preference_words.positive_words,
                negative_words=preference_words.negative_words,
                references=references,
            )

        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "reward_funcs": [reward_func],
        }
        grpo_trainer_params = set(inspect.signature(TRLGRPOTrainer.__init__).parameters.keys())
        if "processing_class" in grpo_trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in grpo_trainer_params:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = TRLGRPOTrainer(**trainer_kwargs)
        trainer.train()

        ckpt_dir = self.output_root / "grpo" / "checkpoint-final"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))

        metadata = {
            "stage": "grpo",
            "train_path": str(train_file),
            "eval_path": eval_path,
            "qlora_checkpoint_dir": str(qlora_ckpt),
            "config": self.config.__dict__,
            "preference_words_size": {
                "positive": len(preference_words.positive_words),
                "negative": len(preference_words.negative_words),
            },
            "artifact": "GRPO aligned adapter checkpoint",
        }
        with (ckpt_dir / "meta.json").open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False, indent=2)

        return str(ckpt_dir)


def _normalize_prompt_text(prompt_item) -> str:
    if isinstance(prompt_item, str):
        return prompt_item
    if isinstance(prompt_item, list):
        parts: list[str] = []
        for seg in prompt_item:
            if isinstance(seg, dict):
                parts.append(str(seg.get("content", "")))
            else:
                parts.append(str(seg))
        return "\n".join(parts)
    return str(prompt_item)


def _normalize_completion_text(completion_item) -> str:
    if isinstance(completion_item, str):
        return completion_item
    if isinstance(completion_item, list):
        parts: list[str] = []
        for seg in completion_item:
            if isinstance(seg, dict):
                parts.append(str(seg.get("content", "")))
            else:
                parts.append(str(seg))
        return "\n".join(parts)
    if isinstance(completion_item, dict):
        return str(completion_item.get("content", ""))
    return str(completion_item)


def _build_grpo_config(config: GRPOTrainConfig, output_dir: str) -> TRLGRPOConfig:
    """兼容不同 TRL 版本的 GRPOConfig 字段差异。"""
    kwargs = {
        "output_dir": output_dir,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.num_train_epochs,
        "num_generations": config.num_generations,
        "beta": config.beta,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "logging_steps": config.logging_steps,
        "report_to": "none",
    }

    signature_params = set(inspect.signature(TRLGRPOConfig.__init__).parameters.keys())
    if "max_completion_length" in signature_params:
        kwargs["max_completion_length"] = config.max_completion_length
    elif "max_new_tokens" in signature_params:
        kwargs["max_new_tokens"] = config.max_completion_length

    return TRLGRPOConfig(**kwargs)
