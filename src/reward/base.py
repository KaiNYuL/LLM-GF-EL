from __future__ import annotations

from typing import Protocol


class RewardEngine(Protocol):
    """奖励引擎协议。

    训练器只依赖该协议，不依赖具体实现，便于后续替换不同奖励策略。
    """

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        positive_words: list[str],
        negative_words: list[str],
        references: list[str] | None = None,
    ) -> list[float]:
        """计算每个 completion 对应的奖励分。"""
        ...
