from __future__ import annotations

import argparse
import json

from src.config.settings import load_settings
from src.data.schemas import PreferenceWords
from src.pipeline.orchestrator import run_qlora_then_grpo


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 QLoRA->GRPO 全流程")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="配置文件路径")
    parser.add_argument("--train", type=str, required=True, help="训练数据 jsonl 路径")
    parser.add_argument("--pref", type=str, required=True, help="偏好词 JSON 路径")
    parser.add_argument("--eval", type=str, default=None, help="评估数据 jsonl 路径")
    args = parser.parse_args()

    settings = load_settings(args.config)
    with open(args.pref, "r", encoding="utf-8") as fp:
        pref = PreferenceWords(**json.load(fp))

    result = run_qlora_then_grpo(settings=settings, train_path=args.train, preference_words=pref, eval_path=args.eval)
    print("Pipeline finished")
    print(f"QLoRA checkpoint: {result.qlora_checkpoint}")
    print(f"GRPO checkpoint: {result.grpo_checkpoint}")


if __name__ == "__main__":
    main()
