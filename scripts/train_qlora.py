from __future__ import annotations

import argparse

from src.config.settings import load_settings
from src.training.qlora_trainer import QLoRATrainConfig, QLoRATrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 QLoRA 训练")
    parser.add_argument("--config", type=str, default="configs/qlora.yaml", help="配置文件路径")
    parser.add_argument("--train", type=str, required=True, help="训练数据 jsonl 路径")
    parser.add_argument("--eval", type=str, default=None, help="评估数据 jsonl 路径")
    args = parser.parse_args()

    settings = load_settings(args.config)
    trainer = QLoRATrainer(QLoRATrainConfig.from_settings(settings.qlora), settings.runtime.output_root)
    ckpt = trainer.train(train_path=args.train, eval_path=args.eval)
    print(f"QLoRA checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
