from unittest.mock import Mock

from src.database.db import SupabaseDB


def make_db():
    config = Mock()
    config.get_supabase_connection.return_value = Mock()
    config.logger = Mock()
    return SupabaseDB(config), config


def test_get_threads_returns_empty_when_user_not_set():
    db, _ = make_db()
    assert db.get_threads() == []


def test_load_chat_history_returns_empty_when_user_not_set():
    db, _ = make_db()
    assert db.load_chat_history("thread-1") == []


def test_save_message_returns_false_when_user_not_set():
    db, _ = make_db()
    assert db.save_message("thread-1", "user", {"text": "hello"}) is False
