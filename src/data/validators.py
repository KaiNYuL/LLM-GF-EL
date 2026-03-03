from __future__ import annotations

from typing import Any

from src.data.schemas import PreferenceWords, TrainSample


def validate_samples(samples: list[TrainSample]) -> tuple[list[TrainSample], list[dict[str, Any]]]:
    """校验样本并过滤不合格数据。

    返回：
    - valid_samples: 可用于训练的样本
    - invalid_reports: 不合格样本报告（含原因）
    """
    valid_samples: list[TrainSample] = []
    invalid_reports: list[dict[str, Any]] = []

    for sample in samples:
        reasons: list[str] = []
        if not sample.id:
            reasons.append("id 为空")
        if sample.type not in {"think", "chat", "rag", "tool"}:
            reasons.append("type 非法")
        if not sample.prompt or not sample.prompt.strip():
            reasons.append("prompt 为空")

        if reasons:
            invalid_reports.append({"id": sample.id, "reasons": reasons})
        else:
            valid_samples.append(sample)

    return valid_samples, invalid_reports


def normalize_preference_words(words: PreferenceWords) -> PreferenceWords:
    """清洗偏好词：去空、去首尾空格、去重。"""

    def _clean(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    return PreferenceWords(
        positive_words=_clean(words.positive_words),
        negative_words=_clean(words.negative_words),
    )
