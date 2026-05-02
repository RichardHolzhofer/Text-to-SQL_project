import argparse
import json
import requests
from requests.auth import HTTPBasicAuth
from pathlib import Path
import yaml
from langfuse import Langfuse
from src.config.config import Config


def sync_prompts():
    """
    CLI utility to sync prompts between local YAML files and Langfuse.
    """
    parser = argparse.ArgumentParser(
        description="Sync prompts between local YAML and Langfuse."
    )
    parser.add_argument(
        "--push", action="store_true", help="Push local YAMLs to Langfuse."
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull production prompts from Langfuse to local YAMLs.",
    )

    args = parser.parse_args()

    # Use central Config
    config = Config()

    if not config.lf_public_key or not config.lf_secret_key:
        print("Error: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env")
        return

    lf = Langfuse(
        public_key=config.lf_public_key,
        secret_key=config.lf_secret_key,
        host=config.lf_host,
    )

    prompt_dir = Path("src/prompts")
    if not prompt_dir.exists():
        print(f"Error: Prompt directory {prompt_dir} does not exist.")
        return

    if args.push:
        print(">>> Pushing local YAMLs to Langfuse...")
        for yaml_file in prompt_dir.glob("*.yaml"):
            name = yaml_file.stem
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

                # Extract messages and metadata
                messages = data.pop("messages", [])

                # Ensure 'node' is in metadata, default to filename
                if "node" not in data:
                    data["node"] = name

                # The rest is config
                metadata = data

                # Check for changes before pushing
                push_needed = True
                try:
                    existing_prompt = lf.get_prompt(name, label="production")

                    def normalize(obj):
                        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

                    existing_clean_msgs = [
                        {k: v for k, v in m.items() if k != "type"}
                        for m in existing_prompt.prompt
                    ]
                    existing_config = existing_prompt.config or {}

                    if normalize(existing_clean_msgs) == normalize(
                        messages
                    ) and normalize(existing_config) == normalize(metadata):
                        push_needed = False
                        print(f"  [SKIPPED] {name} is already up-to-date in Langfuse.")
                except Exception:
                    pass

                if push_needed:
                    print(
                        f"Creating/Updating prompt: {name} (Node: {metadata['node']})..."
                    )
                    lf.create_prompt(
                        name=name,
                        prompt=messages,
                        config=metadata,
                        type="chat",
                        labels=["production"],
                        tags=[metadata["node"]],  # Add node as a tag for UI filtering
                    )
                    print(f"  [SUCCESS] {name} pushed.")
            except Exception as e:
                print(f"  [FAILED] {name}: {e}")

    elif args.pull:
        print(">>> Pulling all production prompts from Langfuse V2 API...")
        try:
            # Using the V2 API as recommended
            api_url = f"{config.lf_host.rstrip('/')}/api/public/v2/prompts"

            # Note: We can pass params if we want to filter, but fetching all is fine
            response = requests.get(
                api_url, auth=HTTPBasicAuth(config.lf_public_key, config.lf_secret_key)
            )

            if response.status_code != 200:
                print(
                    f"Error fetching prompts from V2 API: {response.status_code} - {response.text}"
                )
                return

            all_prompts_data = response.json().get("data", [])

            if not all_prompts_data:
                print("No prompts found in Langfuse.")
                return

            for p_summary in all_prompts_data:
                name = p_summary.get("name")
                if not name:
                    continue
                print(f"Syncing {name}...")

                try:
                    # Fetch the version labeled 'production' using the SDK
                    prompt = lf.get_prompt(name, label="production")

                    # Construct the YAML data
                    new_data = {}

                    # Add metadata from config
                    if prompt.config:
                        new_data.update(prompt.config)

                    # Ensure name and version are included
                    new_data["name"] = name
                    new_data["version"] = prompt.version

                    # Add messages (Langfuse returns list of dicts)
                    raw_messages = prompt.prompt
                    clean_messages = []
                    for msg in raw_messages:
                        # Strip 'type' field if present to keep YAML clean
                        m = {k: v for k, v in msg.items() if k != "type"}
                        clean_messages.append(m)

                    new_data["messages"] = clean_messages

                    # Custom Dumper to force block scalars (|) for multi-line strings
                    def str_presenter(dumper, data):
                        if len(data.splitlines()) > 1:
                            return dumper.represent_scalar(
                                "tag:yaml.org,2002:str", data, style="|"
                            )
                        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

                    yaml.add_representer(str, str_presenter)

                    # Save to YAML file
                    yaml_file = prompt_dir / f"{name}.yaml"
                    with open(yaml_file, "w", encoding="utf-8") as f:
                        yaml.dump(new_data, f, sort_keys=False, allow_unicode=True)

                    print(f"  [SUCCESS] {name} updated to v{prompt.version}.")
                except Exception as e:
                    # It's possible a prompt exists but has no 'production' label
                    if "not found" in str(e).lower():
                        print(f"  [SKIPPED] {name} has no 'production' label.")
                    else:
                        print(f"  [FAILED] {name}: {e}")

        except Exception as e:
            print(f">>> Global pull failed: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    sync_prompts()
