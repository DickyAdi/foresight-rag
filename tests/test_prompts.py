"""Tests that Jinja2 prompt templates render without errors given expected variables."""
import pytest
from foresight.prompts import render


TEMPLATE_VARS = {
    "strips_tools_system": {},
    "baseline_system": {},
    "planner": {
        "question": "Who is the CEO of the company that makes Python?",
        "working_memory": [],
        "beam_width": 3,
        "max_steps": 3,
    },
    "executor": {
        "question": "Who is the CEO of the company that makes Python?",
        "plan": ["retrieve(Python creator)", "retrieve(CEO)"],
        "working_memory": [],
    },
    "answer": {
        "question": "Who is the CEO of the company that makes Python?",
        "working_memory": [{"title": "Python (programming language)", "text": "Python was created by Guido van Rossum."}],
    },
    "reflector": {
        "question": "Who is the CEO of the company that makes Python?",
        "final_answer": "Guido van Rossum",
        "working_memory": [],
    },
}


@pytest.mark.parametrize("template_name,variables", TEMPLATE_VARS.items())
def test_template_renders(template_name, variables):
    result = render(template_name, **variables)
    assert isinstance(result, str)
    assert len(result) > 0
