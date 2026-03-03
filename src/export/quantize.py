from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def quantize_to_gguf(
    model_dir: str,
    out_dir: str,
    quant_type: str = "Q4_K_M",
    llama_cpp_dir: str | None = None,
    model_name: str = "model",
) -> str:
    """将 HF 模型目录转换并量化为 GGUF。

    需要 llama.cpp 环境：
    - convert_hf_to_gguf.py
    - llama-quantize(.exe) 或 quantize(.exe)
    """
    source = Path(model_dir)
    if not source.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    target = Path(out_dir)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    converter = _find_converter_script(llama_cpp_dir)
    quantizer = _find_quantizer_binary(llama_cpp_dir)

    fp16_gguf = target / f"{model_name}.f16.gguf"
    quant_gguf = target / f"{model_name}.{quant_type.lower()}.gguf"

    convert_cmd = [
        sys.executable,
        str(converter),
        str(source),
        "--outfile",
        str(fp16_gguf),
        "--outtype",
        "f16",
    ]
    subprocess.run(convert_cmd, check=True)

    quant_cmd = [str(quantizer), str(fp16_gguf), str(quant_gguf), quant_type]
    subprocess.run(quant_cmd, check=True)

    with (target / "gguf_meta.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "model_dir": model_dir,
                "quant_type": quant_type,
                "format": "GGUF",
                "converter": str(converter),
                "quantizer": str(quantizer),
                "fp16_gguf": str(fp16_gguf),
                "quant_gguf": str(quant_gguf),
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    return str(target)


def _find_converter_script(llama_cpp_dir: str | None) -> Path:
    candidates: list[Path] = []
    if llama_cpp_dir:
        root = Path(llama_cpp_dir)
        candidates.append(root / "convert_hf_to_gguf.py")
    candidates.append(Path("convert_hf_to_gguf.py"))

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("未找到 convert_hf_to_gguf.py，请通过 --llama_cpp_dir 指定 llama.cpp 根目录")


def _find_quantizer_binary(llama_cpp_dir: str | None) -> Path:
    names = ["llama-quantize.exe", "quantize.exe", "llama-quantize", "quantize"]
    if llama_cpp_dir:
        root = Path(llama_cpp_dir)
        for name in names:
            candidate = root / "build" / "bin" / name
            if candidate.exists():
                return candidate
            candidate = root / name
            if candidate.exists():
                return candidate

    for name in names:
        bin_path = shutil.which(name)
        if bin_path:
            return Path(bin_path)

    raise FileNotFoundError("未找到 llama.cpp quantize 可执行文件，请通过 --llama_cpp_dir 指定")
