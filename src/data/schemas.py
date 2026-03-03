from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SampleType = Literal["think", "chat", "rag", "tool"]


class TrainSample(BaseModel):
    """标准训练样本结构。

    字段与项目书保持一致，便于后续 datasets/TRL 直接消费。
    """

    id: str
    type: SampleType
    prompt: str
    reference: str | None = None
    system_prompt: str | None = None


class PreferenceWords(BaseModel):
    """偏好词字典结构。

    两个字段都要求是一维字符串数组（List[str]）。
    """

    positive_words: list[str] = Field(default_factory=list)
    negative_words: list[str] = Field(default_factory=list)
