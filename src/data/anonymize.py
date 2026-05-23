"""Anonimización mínima viable — componente C0 del sistema (TDT §4 RGPD).

Sustituye emails, URLs y menciones por tokens placeholders preservando la
señal estructural (un email apareció aquí) sin filtrar PII al modelo.
"""
from __future__ import annotations

import re


_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_HANDLE = re.compile(r"(?<!\w)@[A-Za-z0-9_-]{2,}")


def anonymize(text: str, *, emails: bool = True, urls: bool = True, handles: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    if urls:
        text = _RE_URL.sub(" <URL> ", text)
    if emails:
        text = _RE_EMAIL.sub(" <EMAIL> ", text)
    if handles:
        text = _RE_HANDLE.sub(" <USER> ", text)
    return text
