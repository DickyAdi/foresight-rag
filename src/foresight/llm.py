"""ChatOpenAI factory functions for each model role via OpenRouter."""
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from foresight.config import (
    OPENROUTER_BASE_URL,
    PLANNER_MODEL,
    EXECUTOR_MODEL,
    REFLECTOR_MODEL,
)

load_dotenv()


def _make_llm(model: str, **kwargs) -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Copy .env.example → .env and fill it in.")
    return ChatOpenAI(model=model, base_url=OPENROUTER_BASE_URL, api_key=api_key, **kwargs)


def get_planner_llm() -> ChatOpenAI:
    return _make_llm(PLANNER_MODEL)


def get_executor_llm() -> ChatOpenAI:
    return _make_llm(EXECUTOR_MODEL)


def get_reflector_llm() -> ChatOpenAI:
    return _make_llm(REFLECTOR_MODEL)
