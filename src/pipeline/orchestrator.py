from __future__ import annotations

from dataclasses import dataclass

from src.config.settings import AppSettings
from src.data.schemas import PreferenceWords
from src.reward.style_reward import StyleRewardEngine
from src.training.grpo_trainer import GRPOTrainConfig, GRPOTrainer
from src.training.qlora_trainer import QLoRATrainConfig, QLoRATrainer


@dataclass
class PipelineResult:
    """流水线执行结果。"""

    qlora_checkpoint: str
    grpo_checkpoint: str


def run_qlora_then_grpo(
    settings: AppSettings,
    train_path: str,
    preference_words: PreferenceWords,
    eval_path: str | None = None,
) -> PipelineResult:
    """串联执行 QLoRA 与 GRPO。

    该函数体现“模块解耦、流程耦合”的设计：
    - 模块上：两个 Trainer 独立；
    - 流程上：GRPO 必须依赖 QLoRA 的 checkpoint。
    """
    qlora_trainer = QLoRATrainer(
        config=QLoRATrainConfig.from_settings(settings.qlora),
        output_root=settings.runtime.output_root,
    )
    qlora_ckpt = qlora_trainer.train(train_path=train_path, eval_path=eval_path)

    reward_engine = StyleRewardEngine()
    grpo_trainer = GRPOTrainer(
        config=GRPOTrainConfig.from_settings(settings.grpo),
        output_root=settings.runtime.output_root,
    )
    grpo_ckpt = grpo_trainer.train(
        train_path=train_path,
        qlora_checkpoint_dir=qlora_ckpt,
        reward_engine=reward_engine,
        preference_words=preference_words,
        eval_path=eval_path,
    )

    return PipelineResult(qlora_checkpoint=qlora_ckpt, grpo_checkpoint=grpo_ckpt)
