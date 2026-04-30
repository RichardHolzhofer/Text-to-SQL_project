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

    def get_sessions(self) -> List[Dict[str, Any]]:
        """Fetches all conversation sessions for the current user."""
        if not self.user_id:
            return []
        try:
            response = (
                self.supabase_conn.table("sessions")
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

    def create_session(self, thread_id: str, title: str = "New Conversation"):
        """Creates a new session entry in the database."""
        if not self.user_id:
            return
        try:
            self.supabase_conn.table("sessions").upsert(
                {"id": thread_id, "user_id": self.user_id, "title": title}
            ).execute()
        except Exception as e:
            error = SupabaseQueryError(e)
            self.config.logger.error(error)

    def load_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Loads the last 50 messages for a specific session and user."""
        if not self.user_id:
            return []

        try:
            response = (
                self.supabase_conn.table("messages")
                .select("*")
                .eq("session_id", session_id)
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
        self, session_id: str, role: str, content: Any, msg_type: str = "text"
    ) -> bool:
        """Saves a single chat message to the database."""
        if not self.user_id:
            return False

        try:
            self.supabase_conn.table("messages").insert(
                {
                    "session_id": session_id,
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

    def create_tables(self):
        """Creates the necessary tables in Supabase using psycopg2."""

        session_table_creation = """
        CREATE TABLE IF NOT EXISTS sessions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES auth.users(id),
        title TEXT,
        created_at TIMESTAMPTZ DEFAULT now()
        );
        """
        messages_table_creation = """
        CREATE TABLE IF NOT EXISTS messages (
        id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
        session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        user_id UUID NOT NULL REFERENCES auth.users(id),
        role TEXT NOT NULL,
        type TEXT NOT NULL DEFAULT 'text',
        content JSONB NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now()
        );
        """
        tasks = [
            (session_table_creation, "sessions"),
            (messages_table_creation, "messages"),
            # Enable RLS
            ("ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;", "enable RLS sessions"),
            ("ALTER TABLE messages ENABLE ROW LEVEL SECURITY;", "enable RLS messages"),
            # Create Policies
            (
                'CREATE POLICY "Users can view their own sessions" ON sessions FOR SELECT USING (auth.uid() = user_id);',
                "policy select sessions",
            ),
            (
                'CREATE POLICY "Users can insert their own sessions" ON sessions FOR INSERT WITH CHECK (auth.uid() = user_id);',
                "policy insert sessions",
            ),
            (
                'CREATE POLICY "Users can update their own sessions" ON sessions FOR UPDATE USING (auth.uid() = user_id);',
                "policy update sessions",
            ),
            (
                'CREATE POLICY "Users can delete their own sessions" ON sessions FOR DELETE USING (auth.uid() = user_id);',
                "policy delete sessions",
            ),
            (
                'CREATE POLICY "Users can view their own messages" ON messages FOR SELECT USING (auth.uid() = user_id);',
                "policy select messages",
            ),
            (
                'CREATE POLICY "Users can insert their own messages" ON messages FOR INSERT WITH CHECK (auth.uid() = user_id);',
                "policy insert messages",
            ),
            (
                'CREATE POLICY "Users can delete their own messages" ON messages FOR DELETE USING (auth.uid() = user_id);',
                "policy delete messages",
            ),
        ]

        for query, name in tasks:
            if name in ["sessions", "messages"]:
                self._execute_query(query, f"Creating {name} table")
            else:
                self._execute_query(query, "Adding policies...")

    def drop_tables(self):
        """Drops the tables from Supabase using psycopg2."""

        messages_table_deletion = "DROP TABLE IF EXISTS messages;"
        sessions_table_deletion = "DROP TABLE IF EXISTS sessions;"
        # LangGraph internal tables
        checkpoint_writes_deletion = "DROP TABLE IF EXISTS checkpoint_writes;"
        checkpoint_blobs_deletion = "DROP TABLE IF EXISTS checkpoint_blobs;"
        checkpoint_migrations_deletion = "DROP TABLE IF EXISTS checkpoint_migrations;"
        checkpoints_deletion = "DROP TABLE IF EXISTS checkpoints;"

        tasks = [
            (messages_table_deletion, "messages"),
            (sessions_table_deletion, "sessions"),
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
