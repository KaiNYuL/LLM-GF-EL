from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_template("【检索上下文】{context} 【用户问题】{question}")


def build_rag_prompt(context: str, question: str) -> str:
    """构造与训练数据一致的 RAG Prompt。"""
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)
