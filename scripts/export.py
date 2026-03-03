from __future__ import annotations

import argparse

from src.export.merger import export_adapter, merge_lora
from src.export.quantize import quantize_to_gguf


def main() -> None:
    parser = argparse.ArgumentParser(description="导出模型")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练产物目录")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-7B-Instruct", help="基座模型")
    parser.add_argument("--out", type=str, default="outputs/export", help="导出目录")
    parser.add_argument("--dtype", type=str, default="float16", help="merge 精度：float16/bfloat16/float32")
    parser.add_argument("--quant_type", type=str, default="Q4_K_M", help="GGUF量化类型")
    parser.add_argument("--llama_cpp_dir", type=str, default=None, help="llama.cpp 根目录（可选）")
    parser.add_argument("--model_name", type=str, default="grpo_tx", help="导出GGUF文件名前缀")
    args = parser.parse_args()

    adapter_dir = export_adapter(args.checkpoint, f"{args.out}/adapter")
    merged_dir = merge_lora(args.base_model, adapter_dir, f"{args.out}/merged", torch_dtype=args.dtype)
    gguf_dir = quantize_to_gguf(
        merged_dir,
        f"{args.out}/gguf",
        quant_type=args.quant_type,
        llama_cpp_dir=args.llama_cpp_dir,
        model_name=args.model_name,
    )
    print(f"Adapter: {adapter_dir}")
    print(f"Merged: {merged_dir}")
    print(f"GGUF: {gguf_dir}")


if __name__ == "__main__":
    main()
