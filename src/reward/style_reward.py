from __future__ import annotations

import re


class StyleRewardEngine:
    """语气偏好奖励引擎。

    默认策略：
    - 命中正向词加分；
    - 命中负向词扣分；
    - 相关性过低惩罚；
    - 长度异常惩罚；
    - 奖励分下限为 0，防止训练崩坏。
    """

    def __init__(
        self,
        positive_weight: float = 0.2,
        negative_weight: float = 0.3,
        max_positive_score: float = 1.0,
        max_negative_penalty: float = 1.0,
        min_relevance: float = 0.3,
        min_relevance_floor: float = 0.6,
    ) -> None:
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.max_positive_score = max_positive_score
        self.max_negative_penalty = max_negative_penalty
        self.min_relevance = min_relevance
        self.min_relevance_floor = min_relevance_floor

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        positive_words: list[str],
        negative_words: list[str],
        references: list[str] | None = None,
    ) -> list[float]:
        rewards: list[float] = []
        for index, completion in enumerate(completions):
            base_score = 1.0

            positive_matches = sum(1 for word in positive_words if word and word in completion)
            positive_score = min(positive_matches * self.positive_weight, self.max_positive_score)

            negative_matches = sum(1 for word in negative_words if word and word in completion)
            negative_penalty = min(negative_matches * self.negative_weight, self.max_negative_penalty)

            relevance_score = self._relevance_score(prompts[index], completion, references[index] if references else None)
            length_score = self._length_score(completion)

            final_reward = (base_score + positive_score - negative_penalty) * relevance_score * length_score
            rewards.append(max(final_reward, 0.0))
        return rewards

    def _relevance_score(self, prompt: str, completion: str, reference: str | None) -> float:
        """中文友好的相关性校验。

        说明：
        - 使用字符级重叠，避免中文 split 后全为空；
        - 低相关时不再直接归零，而是降权到保底分，避免 GRPO 奖励全零。
        """
        source = reference if reference else prompt
        source_chars = _to_char_set(source)
        completion_chars = _to_char_set(completion)
        if not source_chars:
            return 1.0

        overlap = len(source_chars & completion_chars) / len(source_chars)
        if overlap >= self.min_relevance:
            return 1.0
        return max(self.min_relevance_floor, overlap)

    @staticmethod
    def _length_score(completion: str) -> float:
        """长度惩罚，避免极短/极长的无效回复。"""
        length = len(completion)
        if length < 10:
            return 0.2
        if length > 1000:
            return 0.5
        return 1.0


def _to_char_set(text: str) -> set[str]:
    cleaned = re.sub(r"\s+", "", text or "")
    cleaned = re.sub(r"[\W_]+", "", cleaned, flags=re.UNICODE)
    return set(cleaned)
