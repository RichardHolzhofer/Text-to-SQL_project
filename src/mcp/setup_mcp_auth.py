import getpass
import os
import sys

from dotenv import load_dotenv

# Ensure the root of the project is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config.config import Config
from src.database.db import SupabaseDB


def update_env_file(user_id: str, user_email: str):
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
    lines = []

    # Read existing .env if it exists
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    # Filter out existing MCP keys
    new_lines = []
    for line in lines:
        if not (line.startswith("MCP_USER_ID=") or line.startswith("MCP_USER_EMAIL=")):
            new_lines.append(line)

    # Make sure we end with a newline
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    # Append the new MCP credentials
    new_lines.append(f"MCP_USER_ID={user_id}\n")
    new_lines.append(f"MCP_USER_EMAIL={user_email}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"Successfully updated {env_path} with:")
    print(f"  MCP_USER_ID={user_id}")
    print(f"  MCP_USER_EMAIL={user_email}")


def main():
    # Load env vars first
    load_dotenv()

    print("--- Text-to-SQL MCP Authentication Setup ---")
    email = input("Supabase Email: ").strip()
    password = getpass.getpass("Supabase Password: ")

    if not email or not password:
        print("Error: Email and password are required.")
        sys.exit(1)

    try:
        config = Config()
        # Initialize Langfuse via Config to prevent issues
        config.get_langfuse()
        db = SupabaseDB(config)

        print("\nAuthenticating with Supabase...")
        success, message = db.sign_in(email, password)

        if not success:
            print(f"Authentication failed: {message}")
            sys.exit(1)

        print("Authentication successful!")
        update_env_file(db.user_id, db.user_email)

        print("\nRetrieving your active thread histories...")
        threads = db.get_threads()
        if threads:
            print("\nFound the following existing conversations:")
            print(f"{'Thread ID (Copy this)':<38} | {'Title/Question'}")
            print("-" * 80)
            for t in threads[:15]:  # Show last 15 threads
                title = t.get("title") or "Unnamed thread"
                print(f"{t['thread_id']:<38} | {title}")
            if len(threads) > 15:
                print(f"... and {len(threads) - 15} more threads.")
        else:
            print(
                "No existing conversation history found. A new one will start automatically."
            )

        print("\nSetup complete! You can now start the MCP server.")

    except Exception as e:
        print(f"An unexpected error occurred during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
