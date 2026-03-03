from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from src.data.schemas import TrainSample


def load_raw(path: str) -> pd.DataFrame:
    """读取外部原始数据。

    支持 csv/xlsx/xls/json/parquet/jsonl。
    """
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    if suffix == ".json":
        return pd.read_json(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)

    raise ValueError(f"不支持的数据格式: {suffix}")


def to_standard_samples(raw_df: pd.DataFrame) -> list[TrainSample]:
    """将 DataFrame 映射到标准训练样本列表。"""
    records = raw_df.to_dict(orient="records")
    samples: list[TrainSample] = []
    for item in records:
        sample = TrainSample(
            id=str(item.get("id", "")),
            type=item.get("type", "chat"),
            prompt=str(item.get("prompt", "")),
            reference=item.get("reference"),
            system_prompt=item.get("system_prompt"),
        )
        samples.append(sample)
    return samples


def save_jsonl(samples: list[TrainSample], out_path: str) -> None:
    """保存标准样本到 jsonl 文件。"""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fp:
        for sample in samples:
            fp.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")


def load_streaming_jsonl(path: str):
    """通过 datasets 的 streaming 模式加载 jsonl。

    适用于大数据量场景，避免一次性占满内存。
    """
    return load_dataset("json", data_files=path, split="train", streaming=True)
