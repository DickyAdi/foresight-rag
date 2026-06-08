"""Shared secret loading for adapters.

`load_dotenv()` runs once on first import (any adapter import pulls this in).
`require_api_key()` fetches OPENROUTER_API_KEY with a clear error if it is missing,
so a forgotten `.env` fails loudly at construction rather than as an opaque 401.
Secrets live in `.env` (gitignored) — never hardcode. See design/09-models-and-config.md.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()


def require_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key
