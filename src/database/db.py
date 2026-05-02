import psycopg2
from typing import List, Dict, Any, Tuple, Optional
from supabase import Client
from src.config.config import Config
from src.exceptions.exception import (
    SupabaseAuthError,
    SupabaseConnectionError,
    SupabaseQueryError,
)


class SupabaseDB:
    """
    Manages interactions with Supabase for authentication and chat history persistence.
    """

    def __init__(self, config: Config, admin: bool = False):
        self.config = config
        self.supabase_conn: Client = config.get_supabase_connection(write_access=admin)
        self.user_id: Optional[str] = None
        self.user_email: Optional[str] = None

    def sign_in(self, email: str, password: str) -> Tuple[bool, str]:
        """Authenticates a user with email and password."""
        try:
            response = self.supabase_conn.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            if response.user:
                self.user_id = response.user.id
                self.user_email = response.user.email
                return True, f"Successfully logged in as {response.user.email}!"
            return False, "Login failed: No user returned."
        except Exception as e:
            error = SupabaseAuthError(e)
            self.config.logger.error(error)
            return False, f"Login failed: {str(e)}"

    def sign_up(self, email: str, password: str) -> Tuple[bool, str]:
        """Registers a new user with email and password."""
        try:
            response = self.supabase_conn.auth.sign_up(
                {"email": email, "password": password}
            )
            if response.user:
                return (
                    True,
                    f"Registration successful! Please check your email ({response.user.email}) or log in.",
                )
            return False, "Sign up failed: No user returned."
        except Exception as e:
            error = SupabaseAuthError(e)
            self.config.logger.error(error)
            return False, f"Sign up failed: {str(e)}"

    def sign_out(self):
        """Logs out the current user."""
        try:
            self.supabase_conn.auth.sign_out()
            self.user_id = None
        except Exception:
            pass

    def get_threads(self) -> List[Dict[str, Any]]:
        """Fetches all conversation threads for the current user."""
        if not self.user_id:
            return []
        try:
            response = (
                self.supabase_conn.table("threads")
                .select("*")
                .eq("user_id", self.user_id)
                .order("created_at", desc=True)
                .execute()
            )
            return response.data
        except Exception as e:
            error = SupabaseQueryError(e)
            self.config.logger.error(error)
            return []

    def create_thread(self, thread_id: str, title: str = "New Conversation"):
        """Creates a new thread entry in the database."""
        if not self.user_id:
            return
        try:
            self.supabase_conn.table("threads").upsert(
                {"thread_id": thread_id, "user_id": self.user_id, "title": title}
            ).execute()
        except Exception as e:
            error = SupabaseQueryError(e)
            self.config.logger.error(error)

    def load_chat_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Loads the last 50 messages for a specific thread and user."""
        if not self.user_id:
            return []

        try:
            response = (
                self.supabase_conn.table("messages")
                .select("*")
                .eq("thread_id", thread_id)
                .eq("user_id", self.user_id)
                .order("created_at", desc=False)
                .limit(50)
                .execute()
            )

            return [
                {
                    "role": m["role"],
                    "type": m.get("type", "text"),
                    "content": m["content"],
                }
                for m in response.data
            ]
        except Exception as e:
            error = SupabaseQueryError(e)
            self.config.logger.error(error)
            return []

    def save_message(
        self, thread_id: str, role: str, content: Any, msg_type: str = "text"
    ) -> bool:
        """Saves a single chat message to the database."""
        if not self.user_id:
            return False

        try:
            self.supabase_conn.table("messages").insert(
                {
                    "thread_id": thread_id,
                    "user_id": self.user_id,
                    "role": role,
                    "type": msg_type,
                    "content": content,
                }
            ).execute()
            return True
        except Exception as e:
            error = SupabaseQueryError(e)
            self.config.logger.error(error)
            return False

    def delete_thread(self, thread_id: str) -> bool:
        """
        Deletes a specific thread, its messages, and its LangGraph checkpoints.
        """
        try:
            # 1. Delete LangGraph checkpoints manually from Postgres
            for table in ["checkpoints", "checkpoint_blobs", "checkpoint_writes"]:
                query = f"DELETE FROM {table} WHERE thread_id = '{thread_id}';"
                self._execute_query(
                    query, f"Cleaning up {table} for thread {thread_id}"
                )

            # 2. Delete from threads table (cascades to messages)
            self.supabase_conn.table("threads").delete().eq(
                "thread_id", thread_id
            ).execute()
            return True
        except Exception as e:
            self.config.logger.error(f"Failed to delete thread {thread_id}: {e}")
            return False

    def delete_user(self, user_id: str) -> bool:
        """
        Deletes a user and ALL their associated data (history + checkpoints).
        Requires 'admin=True' (service_role key).
        """
        try:
            # 1. Get all thread_ids for this user to clean up LangGraph memory
            # Note: We must do this before deleting the user/threads
            threads = self.get_threads()
            thread_ids = [t["thread_id"] for t in threads]

            # 2. Delete LangGraph checkpoints manually
            if thread_ids:
                thread_ids_str = ", ".join([f"'{tid}'" for tid in thread_ids])
                for table in ["checkpoints", "checkpoint_blobs", "checkpoint_writes"]:
                    query = (
                        f"DELETE FROM {table} WHERE thread_id IN ({thread_ids_str});"
                    )
                    self._execute_query(query, f"Cleaning up {table}")

            # 3. Delete from Supabase Auth (This cascades to 'threads' and 'messages' tables in public schema)
            # This requires service_role key
            self.supabase_conn.auth.admin.delete_user(user_id)
            return True
        except Exception as e:
            self.config.logger.error(f"Failed to delete user {user_id}: {e}")
            return False

    def insert_schema_cache(self, key: str, content: Any) -> bool:
        """Inserts a new schema version into the history. Requires admin=True."""
        try:
            # Using Supabase client for insert (creates a new row)
            self.supabase_conn.table("schema_cache").insert(
                {"key": key, "content": content}
            ).execute()
            return True
        except Exception as e:
            self.config.logger.error(f"Failed to insert schema cache: {e}")
            return False

    def get_schema_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves a schema context from the cache."""
        try:
            response = (
                self.supabase_conn.table("schema_cache")
                .select("content, updated_at")
                .eq("key", key)
                .order("updated_at", desc=True)
                .limit(1)
                .execute()
            )
            if response.data:
                content = response.data[0]["content"]
                # Inject the updated_at from the table into the schema dict
                content["updated_at"] = response.data[0]["updated_at"]
                return content
            return None
        except Exception as e:
            # We don't log a full error here as it's common for cache to be empty on first run
            self.config.logger.info(f"Schema cache miss for key '{key}': {e}")
            return None

    def create_tables(self):
        """Creates the necessary tables in Supabase using psycopg2."""

        thread_table_creation = """
        CREATE TABLE IF NOT EXISTS threads (
        thread_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES auth.users(id),
        title TEXT,
        created_at TIMESTAMPTZ DEFAULT now()
        );
        """
        messages_table_creation = """
        CREATE TABLE IF NOT EXISTS messages (
        id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
        thread_id UUID NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
        user_id UUID NOT NULL REFERENCES auth.users(id),
        role TEXT NOT NULL,
        type TEXT NOT NULL DEFAULT 'text',
        content JSONB NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now()
        );
        """
        schema_cache_table_creation = """
        CREATE TABLE IF NOT EXISTS schema_cache (
        id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
        key TEXT NOT NULL, -- Removed UNIQUE to allow history
        content JSONB NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now()
        );
        """
        tasks = [
            (thread_table_creation, "threads"),
            (messages_table_creation, "messages"),
            (schema_cache_table_creation, "schema_cache"),
            # Enable RLS
            ("ALTER TABLE threads ENABLE ROW LEVEL SECURITY;", "enable RLS threads"),
            ("ALTER TABLE messages ENABLE ROW LEVEL SECURITY;", "enable RLS messages"),
            (
                "ALTER TABLE schema_cache ENABLE ROW LEVEL SECURITY;",
                "enable RLS schema_cache",
            ),
            # Create Policies
            (
                'DROP POLICY IF EXISTS "Users can view their own threads" ON threads; CREATE POLICY "Users can view their own threads" ON threads FOR SELECT USING (auth.uid() = user_id);',
                "policy select threads",
            ),
            (
                'DROP POLICY IF EXISTS "Users can insert their own threads" ON threads; CREATE POLICY "Users can insert their own threads" ON threads FOR INSERT WITH CHECK (auth.uid() = user_id);',
                "policy insert threads",
            ),
            (
                'DROP POLICY IF EXISTS "Users can update their own threads" ON threads; CREATE POLICY "Users can update their own threads" ON threads FOR UPDATE USING (auth.uid() = user_id);',
                "policy update threads",
            ),
            (
                'DROP POLICY IF EXISTS "Users can delete their own threads" ON threads; CREATE POLICY "Users can delete their own threads" ON threads FOR DELETE USING (auth.uid() = user_id);',
                "policy delete threads",
            ),
            (
                'DROP POLICY IF EXISTS "Users can view their own messages" ON messages; CREATE POLICY "Users can view their own messages" ON messages FOR SELECT USING (auth.uid() = user_id);',
                "policy select messages",
            ),
            (
                'DROP POLICY IF EXISTS "Users can insert their own messages" ON messages; CREATE POLICY "Users can insert their own messages" ON messages FOR INSERT WITH CHECK (auth.uid() = user_id);',
                "policy insert messages",
            ),
            (
                'DROP POLICY IF EXISTS "Users can delete their own messages" ON messages; CREATE POLICY "Users can delete their own messages" ON messages FOR DELETE USING (auth.uid() = user_id);',
                "policy delete messages",
            ),
            (
                'DROP POLICY IF EXISTS "Authenticated users can view schema cache" ON schema_cache; CREATE POLICY "Authenticated users can view schema cache" ON schema_cache FOR SELECT TO authenticated USING (true);',
                "policy select schema_cache",
            ),
        ]

        for query, name in tasks:
            if name in ["threads", "messages", "schema_cache"]:
                self._execute_query(query, f"Creating {name} table")
            else:
                self._execute_query(query, "Adding policies...")

    def drop_tables(self):
        """Drops the tables from Supabase using psycopg2."""

        messages_table_deletion = "DROP TABLE IF EXISTS messages;"
        threads_table_deletion = "DROP TABLE IF EXISTS threads;"
        schema_cache_deletion = "DROP TABLE IF EXISTS schema_cache;"
        # LangGraph internal tables
        checkpoint_writes_deletion = "DROP TABLE IF EXISTS checkpoint_writes;"
        checkpoint_blobs_deletion = "DROP TABLE IF EXISTS checkpoint_blobs;"
        checkpoint_migrations_deletion = "DROP TABLE IF EXISTS checkpoint_migrations;"
        checkpoints_deletion = "DROP TABLE IF EXISTS checkpoints;"

        tasks = [
            (messages_table_deletion, "messages"),
            (threads_table_deletion, "threads"),
            (schema_cache_deletion, "schema_cache"),
            (checkpoint_writes_deletion, "checkpoint_writes"),
            (checkpoint_blobs_deletion, "checkpoint_blobs"),
            (checkpoint_migrations_deletion, "checkpoint_migrations"),
            (checkpoints_deletion, "checkpoints"),
        ]

        for query, name in tasks:
            self._execute_query(query, f"Dropping {name} table")

    def _execute_query(self, query: str, action_name: str):
        """Helper to execute a single query using psycopg2."""
        conn = None
        try:
            conn = psycopg2.connect(self.config.sb_db_uri)
            with conn.cursor() as cursor:
                cursor.execute(query)
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise SupabaseConnectionError(f"{action_name} failed: {str(e)}")
        finally:
            if conn:
                conn.close()
