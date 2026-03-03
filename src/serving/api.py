from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="grpo-tx-model")
    messages: list[ChatMessage]
    temperature: float = 0.7


def create_app(model_dir: str) -> FastAPI:
    """创建 FastAPI 服务。

    注意：这里提供 OpenAI 风格接口骨架，方便先接入 Agent。
    真实推理可在路由中替换为 transformers/vllm 推理逻辑。
    """
    app = FastAPI(title="GRPO-TX Serving API")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_dir": model_dir}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> dict:
        user_text = req.messages[-1].content if req.messages else ""
        return {
            "id": "chatcmpl-demo",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[DEMO] 已接收：{user_text}",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    return app
