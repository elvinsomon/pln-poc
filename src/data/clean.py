"""Limpieza textual ligera, componible.

Diseñada para preservar señales lingüísticas relevantes (errores ortográficos,
fragmentariedad, registro emocional) identificadas en el TDT §2.2; solo elimina
ruido estructural (markdown, fences de código) que distorsiona TF-IDF.
"""
from __future__ import annotations

import re


_RE_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_RE_INLINE_CODE = re.compile(r"`[^`\n]+`")
_RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_RE_MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_MULTISPACE = re.compile(r"\s+")


def strip_code_blocks(text: str) -> str:
    return _RE_CODE_FENCE.sub(" <CODE> ", _RE_INLINE_CODE.sub(" <CODE> ", text))


def strip_markdown(text: str) -> str:
    text = _RE_MD_IMG.sub(" ", text)
    text = _RE_MD_LINK.sub(r"\1", text)
    return _RE_HTML_TAG.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    return _RE_MULTISPACE.sub(" ", text).strip()


def truncate(text: str, max_chars: int) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def clean_text(text: str, *, max_chars: int | None = None,
               do_code: bool = True, do_md: bool = True, do_ws: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    if do_code:
        text = strip_code_blocks(text)
    if do_md:
        text = strip_markdown(text)
    if do_ws:
        text = normalize_whitespace(text)
    if max_chars is not None:
        text = truncate(text, max_chars)
    return text
