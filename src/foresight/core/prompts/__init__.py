"""Jinja2 prompt loader. All prompt text lives in *.jinja2 files in this directory.
No prompt strings are hardcoded in Python — code only passes variables.

Conventions (see design/12-implementation-architecture.md):
  - PackageLoader (importlib-based) so prompts resolve when pip-installed.
  - One template = one message role: *_system.jinja2 -> system, others -> human.
  - Shared fragments via {% include %} (e.g. the STRIPS tool block).
  - With structured output, templates describe the TASK, not the JSON schema
    (the Pydantic model owns the schema).

Usage:
    from foresight.core.prompts import render
    system_prompt = render("strips_tools_system")
    planner_prompt = render("planner", question=q, working_memory=wm)
"""
from jinja2 import Environment, PackageLoader, StrictUndefined

_env = Environment(
    loader=PackageLoader("foresight.core", "prompts"),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render(template_name: str, **variables: object) -> str:
    return _env.get_template(f"{template_name}.jinja2").render(**variables)
