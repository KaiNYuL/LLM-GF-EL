from __future__ import annotations

import argparse
import json

from src.config.settings import load_settings
from src.data.schemas import PreferenceWords
from src.reward.style_reward import StyleRewardEngine
from src.training.grpo_trainer import GRPOTrainConfig, GRPOTrainer


def _load_preference_words(path: str) -> PreferenceWords:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return PreferenceWords(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 GRPO 训练")
    parser.add_argument("--config", type=str, default="configs/grpo.yaml", help="配置文件路径")
    parser.add_argument("--train", type=str, required=True, help="训练数据 jsonl 路径")
    parser.add_argument("--qlora_ckpt", type=str, required=True, help="QLoRA checkpoint 路径")
    parser.add_argument("--pref", type=str, required=True, help="偏好词 JSON 路径")
    parser.add_argument("--eval", type=str, default=None, help="评估数据 jsonl 路径")
    args = parser.parse_args()

    settings = load_settings(args.config)
    trainer = GRPOTrainer(GRPOTrainConfig.from_settings(settings.grpo), settings.runtime.output_root)
    reward = StyleRewardEngine()
    pref_words = _load_preference_words(args.pref)
    ckpt = trainer.train(
        train_path=args.train,
        qlora_checkpoint_dir=args.qlora_ckpt,
        reward_engine=reward,
        preference_words=pref_words,
        eval_path=args.eval,
    )
    print(f"GRPO checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
