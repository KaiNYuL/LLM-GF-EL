from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def export_adapter(checkpoint_dir: str, out_dir: str) -> str:
    """导出 LoRA 适配器目录。

    这里会复制 checkpoint 中的 LoRA 相关文件到独立目录，方便分发与版本管理。
    """
    source = Path(checkpoint_dir)
    if not source.exists():
        raise FileNotFoundError(f"checkpoint 目录不存在: {checkpoint_dir}")

    target = Path(out_dir)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    for file in source.iterdir():
        if file.is_file():
            shutil.copy2(file, target / file.name)
            copied_files.append(file.name)

    with (target / "adapter_export_meta.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "from": str(source),
                "to": str(target),
                "type": "adapter",
                "files": copied_files,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    return str(target)


def merge_lora(base_model: str, adapter_dir: str, out_dir: str, torch_dtype: str = "float16") -> str:
    """执行真实 LoRA 合并并导出完整模型。"""
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter 目录不存在: {adapter_dir}")

    dtype = _resolve_dtype(torch_dtype)

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    merged_model = model.merge_and_unload()

    target = Path(out_dir)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(str(target), safe_serialization=True)
    tokenizer.save_pretrained(str(target))

    with (target / "merge_meta.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "base_model": base_model,
                "adapter_dir": adapter_dir,
                "type": "merged",
                "torch_dtype": torch_dtype,
                "output_dir": str(target),
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    return str(target)


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32
