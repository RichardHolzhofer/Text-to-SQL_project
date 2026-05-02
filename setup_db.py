import argparse

from src.config.config import Config
from src.database.db import SupabaseDB


def setup_db():
    parser = argparse.ArgumentParser(description="Supabase Database Setup Tool")
    parser.add_argument(
        "--action",
        choices=["create", "drop", "reset"],
        default="create",
        help="Action to perform: create tables, drop tables, or reset (drop then create).",
    )

    args = parser.parse_args()
    config = Config()
    db = SupabaseDB(config, admin=True)

    if args.action == "drop":
        print("Dropping tables...")
        db.drop_tables()
        print("Tables dropped.")

    elif args.action == "create":
        print("Creating tables...")
        db.create_tables()
        print("Tables created successfully.")

    elif args.action == "reset":
        print("Resetting database...")
        db.drop_tables()
        db.create_tables()
        print("Database reset complete.")


if __name__ == "__main__":
    setup_db()
