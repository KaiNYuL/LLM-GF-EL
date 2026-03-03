from __future__ import annotations

import json

from src.eval.runner import EvalInput, run_offline_eval


def main() -> None:
    demo = EvalInput(
        outputs=["哈哈哈这题我会", "我不太确定"],
        positive_words=["哈哈哈", "加油"],
        negative_words=["官方话术"],
        base_answers=["A", "B"],
        tuned_answers=["A", "C"],
        labels=["A", "B"],
        tool_traces=[{"tool": "search", "success": True}, {"tool": "calc", "success": False}],
    )
    metrics = run_offline_eval(demo)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
