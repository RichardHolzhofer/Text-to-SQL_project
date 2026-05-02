import argparse
import json
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

from src.config.config import Config
from src.exceptions.exception import PromptSyncError
from src.utils.utils import load_yaml, save_yaml


class PromptSyncer:
    """
    Utility to sync prompts between local YAML files and Langfuse.
    """

    def __init__(self):
        try:
            self.config = Config(logger_name="prompt-sync")
            self.logger = self.config.logger
            self.prompt_dir = Path("src/prompts")

            if not self.config.lf_public_key or not self.config.lf_secret_key:
                raise ValueError(
                    "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env"
                )

            self.lf = self.config.get_langfuse()
            self.api_url = f"{self.config.lf_host.rstrip('/')}/api/public/v2/prompts"
            self.auth = HTTPBasicAuth(
                self.config.lf_public_key, self.config.lf_secret_key
            )
        except Exception as e:
            raise PromptSyncError(e)

    def _normalize(self, obj):
        """Consistent JSON serialization for comparison."""
        try:
            return json.dumps(obj, sort_keys=True, separators=(",", ":"))
        except Exception as e:
            raise PromptSyncError(e)

    def _is_up_to_date(self, name, local_messages, local_config):
        """Checks if the local prompt matches the production version in Langfuse."""
        try:
            remote = self.lf.get_prompt(name, label="production")
            remote_messages = [
                {k: v for k, v in m.items() if k != "type"} for m in remote.prompt
            ]
            remote_config = remote.config or {}

            messages_match = self._normalize(remote_messages) == self._normalize(
                local_messages
            )
            config_match = self._normalize(remote_config) == self._normalize(
                local_config
            )

            return messages_match and config_match
        except Exception:
            # If prompt doesn't exist or has no production label, it's not up-to-date
            return False

    def push_prompts(self):
        """Pushes local YAML files to Langfuse."""
        try:
            if not self.prompt_dir.exists():
                raise FileNotFoundError(
                    f"Prompt directory {self.prompt_dir} does not exist."
                )

            self.logger.info(">>> Pushing local YAMLs to Langfuse...")

            for yaml_file in self.prompt_dir.glob("*.yaml"):
                name = yaml_file.stem
                try:
                    data = load_yaml(yaml_file)

                    messages = data.pop("messages", [])
                    # Ensure 'node' is in metadata, default to filename
                    if "node" not in data:
                        data["node"] = name

                    if self._is_up_to_date(name, messages, data):
                        self.logger.info(f"  [SKIPPED] {name} is already up-to-date.")
                        continue

                    self.logger.info(
                        f"  [PUSHING] {name} (Node: {data.get('node')})..."
                    )
                    self.lf.create_prompt(
                        name=name,
                        prompt=messages,
                        config=data,
                        type="chat",
                        labels=["production"],
                        tags=[data["node"]],
                    )
                    self.logger.info(f"  [SUCCESS] {name} pushed.")
                except Exception as e:
                    self.logger.error(PromptSyncError(e))
        except Exception as e:
            raise PromptSyncError(e)

    def pull_prompts(self):
        """Pulls production prompts from Langfuse to local YAML files."""
        self.logger.info(">>> Pulling production prompts from Langfuse...")

        try:
            response = requests.get(
                self.api_url,
                auth=self.auth,
            )

            if response.status_code != 200:
                raise PromptSyncError(
                    RuntimeError(f"API Error: {response.status_code} - {response.text}")
                )

            prompts_list = response.json().get("data", [])
            if not prompts_list:
                self.logger.info("No prompts found in Langfuse.")
                return

            for p_summary in prompts_list:
                name = p_summary.get("name")
                if not name:
                    continue

                try:
                    prompt = self.lf.get_prompt(name, label="production")
                    prompt_data = dict(prompt.config or {})
                    prompt_data["name"] = name
                    prompt_data["version"] = prompt.version
                    prompt_data["messages"] = [
                        {k: v for k, v in m.items() if k != "type"}
                        for m in prompt.prompt
                    ]

                    yaml_file = self.prompt_dir / f"{name}.yaml"
                    save_yaml(prompt_data, yaml_file)

                    self.logger.info(f"  [SUCCESS] {name} pulled (v{prompt.version}).")
                except Exception as e:
                    if "not found" in str(e).lower():
                        self.logger.info(
                            f"  [SKIPPED] {name} has no 'production' label."
                        )
                    else:
                        self.logger.error(PromptSyncError(e))

        except Exception as e:
            self.logger.error(PromptSyncError(e))


def sync_prompts():
    parser = argparse.ArgumentParser(
        description="Sync prompts between local YAML and Langfuse."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--push", action="store_true", help="Push local YAMLs to Langfuse."
    )
    group.add_argument(
        "--pull",
        action="store_true",
        help="Pull production prompts from Langfuse.",
    )

    args = parser.parse_args()

    try:
        syncer = PromptSyncer()
        if args.push:
            syncer.push_prompts()
        else:
            syncer.pull_prompts()
    except Exception as e:
        # Wrap unknown exceptions to get detailed formatting
        detailed_error = PromptSyncError(e)
        if "syncer" in locals() and hasattr(syncer, "logger"):
            syncer.logger.error(detailed_error)
        else:
            print(f"CRITICAL: {detailed_error}")


if __name__ == "__main__":
    sync_prompts()
