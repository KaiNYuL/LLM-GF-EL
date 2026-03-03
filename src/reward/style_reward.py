from __future__ import annotations


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
    ) -> None:
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.max_positive_score = max_positive_score
        self.max_negative_penalty = max_negative_penalty
        self.min_relevance = min_relevance

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
        """简单关键词重叠相关性校验。

        说明：这是一个轻量默认实现，可在后续替换成 embedding 相似度实现。
        """
        source = reference if reference else prompt
        prompt_keywords = set(source.replace("？", " ").replace("。", " ").split())
        completion_keywords = set(completion.split())
        if not prompt_keywords:
            return 1.0

        overlap = len(prompt_keywords & completion_keywords) / len(prompt_keywords)
        return 1.0 if overlap >= self.min_relevance else 0.0

    @staticmethod
    def _length_score(completion: str) -> float:
        """长度惩罚，避免极短/极长的无效回复。"""
        length = len(completion)
        if length < 10:
            return 0.2
        if length > 1000:
            return 0.5
        return 1.0
