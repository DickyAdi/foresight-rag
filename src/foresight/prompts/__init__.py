"""Jinja2 prompt loader. All prompt text lives in *.jinja2 files in this directory.
No prompt strings are hardcoded in Python — code only passes variables.

Usage:
    from foresight.prompts import render
    system_prompt = render("strips_tools_system")
    planner_prompt = render("planner", question=q, working_memory=wm)
"""
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render(template_name: str, **variables: object) -> str:
    return _env.get_template(f"{template_name}.jinja2").render(**variables)
