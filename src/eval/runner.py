from __future__ import annotations

from dataclasses import dataclass

from src.eval.metrics import rag_accuracy_drop, taboo_hit_rate, tone_match_rate, tool_call_success_rate


@dataclass
class EvalInput:
    """评估输入结构。"""

    outputs: list[str]
    positive_words: list[str]
    negative_words: list[str]
    base_answers: list[str]
    tuned_answers: list[str]
    labels: list[str]
    tool_traces: list[dict]


def run_offline_eval(data: EvalInput) -> dict[str, float]:
    """执行离线评估并返回指标字典。"""
    return {
        "tone_match_rate": tone_match_rate(data.outputs, data.positive_words),
        "taboo_hit_rate": taboo_hit_rate(data.outputs, data.negative_words),
        "rag_accuracy_drop": rag_accuracy_drop(data.base_answers, data.tuned_answers, data.labels),
        "tool_call_success_rate": tool_call_success_rate(data.tool_traces),
    }
