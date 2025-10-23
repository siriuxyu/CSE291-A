from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from agent.utils import load_chat_model

model = "anthropic/claude-sonnet-4-5-20250929"

class TranslateInput(BaseModel):
    """Arguments for the translator tool."""
    text: str = Field(..., description="Source text to translate.")
    target_lang: str = Field(..., description="Target language code or name (e.g., 'en', 'Japanese').")
    source_lang: Optional[str] = Field(None, description="Source language (auto-detect if omitted).")


SYSTEM_TMPL = (
    "You are a translation engine.\n"
    "- Translate the user's TEXT into the TARGET language{src_hint}.\n"
    "- Be faithful to meaning and tone.\n"
    "- Keep inline formatting (Markdown/HTML/code) as-is.\n"
    "- Do not add explanations; output only the translated text."
)


@tool("translator", args_schema=TranslateInput, return_direct=False)
def translator(
    text: str,
    target_lang: str,
    source_lang: Optional[str] = None,
) -> str:
    """
    Minimal translator tool that calls a dedicated LLM to translate text.
    Returns only the translated text.
    """
    src_hint = f" from the SOURCE language '{source_lang}'" if source_lang else " (auto-detect source language)"
    system = SYSTEM_TMPL.format(src_hint=src_hint)

    chat = load_chat_model(model)
    resp = chat.invoke([
        SystemMessage(content=system.replace("TARGET", target_lang)),
        HumanMessage(content=text),
    ])

    content = (resp.content or "").strip()
    return content or "Translator tool error: empty response."
