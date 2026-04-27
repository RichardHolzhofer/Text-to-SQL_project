import yaml
from jinja2 import Template
from typing import Any, Dict


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML file from a string path.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(data: Any) -> str:
    """
    Convert a dictionary/object to a YAML string.
    """
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def load_prompt(path: str, variables: dict):
    """
    Load a prompt YAML and render its content using Jinja2 templates.
    """
    prompt = load_yaml(path)

    messages = []
    for msg in prompt.get("messages", []):
        template = Template(msg["content"])
        rendered = template.render(**variables)

        messages.append((msg["role"], rendered))

    return messages
