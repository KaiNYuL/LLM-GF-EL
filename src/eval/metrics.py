from __future__ import annotations


def tone_match_rate(outputs: list[str], positive_words: list[str]) -> float:
    """语气匹配率：至少命中一个正向词的输出占比。"""
    if not outputs:
        return 0.0
    matched = sum(1 for text in outputs if any(word in text for word in positive_words if word))
    return matched / len(outputs)


def taboo_hit_rate(outputs: list[str], negative_words: list[str]) -> float:
    """禁忌词命中率：命中任一负向词的输出占比。"""
    if not outputs:
        return 0.0
    hit = sum(1 for text in outputs if any(word in text for word in negative_words if word))
    return hit / len(outputs)


def rag_accuracy_drop(base_answers: list[str], tuned_answers: list[str], labels: list[str]) -> float:
    """RAG 准确率下降（简化版本）。

    规则：使用字符串精确匹配 labels，计算微调前后准确率差值（base - tuned）。
    """
    if not labels:
        return 0.0

    base_correct = sum(1 for pred, label in zip(base_answers, labels) if pred == label)
    tuned_correct = sum(1 for pred, label in zip(tuned_answers, labels) if pred == label)
    base_acc = base_correct / len(labels)
    tuned_acc = tuned_correct / len(labels)
    return base_acc - tuned_acc


def tool_call_success_rate(traces: list[dict]) -> float:
    """工具调用成功率。

    trace 示例：{"tool": "search", "success": True}
    """
    if not traces:
        return 0.0
    ok = sum(1 for item in traces if bool(item.get("success")))
    return ok / len(traces)
