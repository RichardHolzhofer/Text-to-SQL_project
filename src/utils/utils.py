import yaml
from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML file from a string path.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(data: dict):
    """
    Convert a dictionary/object to a YAML string.
    """
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def get_prompt_template(prompt_name: str, config=None):
    """
    Fetches a ChatPromptTemplate from Langfuse or local YAML.
    The template carries its own metadata and config (model, temperature, etc.).
    """
    # 1. Attempt Langfuse Fetch
    if config:
        try:
            lf = config.get_langfuse()
            # Fetch the version labeled 'production'
            prompt = lf.get_prompt(prompt_name, label="production")

            # Create LangChain template directly from Langfuse object
            template = ChatPromptTemplate.from_messages(prompt.get_langchain_prompt())

            # Attach metadata for tracing and dynamic overrides
            template.metadata = {
                "langfuse_prompt": prompt,
                "config": prompt.config or {},
            }
            return template
        except Exception as e:
            if hasattr(config, "logger"):
                config.logger.warning(
                    f"Langfuse prompt '{prompt_name}' failed: {str(e)}. Using local fallback."
                )

    # 2. Local Fallback
    if not prompt_name.endswith(".yaml") and "/" not in prompt_name:
        path = f"src/prompts/{prompt_name}.yaml"
    else:
        path = prompt_name

    data = load_yaml(path)

    # We use template_format="jinja2" to support our local {{ variable }} syntax
    template = ChatPromptTemplate.from_messages(
        data.get("messages", []), template_format="jinja2"
    )

    # Attach config for local parity
    template.metadata = {"config": data}

    return template
