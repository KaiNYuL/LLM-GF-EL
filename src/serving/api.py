from __future__ import annotations

import uuid

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="grpo-tx-model")
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 256


def create_app(model_dir: str) -> FastAPI:
    """创建 FastAPI 服务并加载本地模型。"""
    app = FastAPI(title="GRPO-TX Serving API")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_dir": model_dir}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> dict:
        prompt_text = _build_prompt(tokenizer, req.messages)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=True,
                temperature=req.temperature,
                top_p=req.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def _build_prompt(tokenizer: AutoTokenizer, messages: list[ChatMessage]) -> str:
    """构造推理输入文本，优先使用 chat template。"""
    chat_payload = [{"role": msg.role, "content": msg.content} for msg in messages]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(chat_payload, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    lines: list[str] = []
    for msg in messages:
        lines.append(f"{msg.role}: {msg.content}")
    lines.append("assistant:")
    return "\n".join(lines)
