from unittest.mock import Mock, call, patch

import pytest

from src.database.db import SupabaseDB
from src.exceptions.exception import (
    ConfigError,
    NetworkRetryExhaustedError,
    SupabaseConnectionError,
)


def test_init_success(config, mock_supabase_factory):
    _, mock_supabase_client = mock_supabase_factory

    db = SupabaseDB(config)

    assert db.config == config
    assert db.user_id is None
    assert db.user_email is None
    assert db.supabase_conn == mock_supabase_client


def test_init_admin_connection(config):
    mock_supabase_conn = Mock()

    config.get_supabase_connection = Mock(return_value=mock_supabase_conn)

    db = SupabaseDB(config, admin=True)
    assert db.supabase_conn == mock_supabase_conn

    config.get_supabase_connection.assert_called_once_with(write_access=True)


def test_init_reraises_config_error(config):
    config.get_supabase_connection = Mock(side_effect=ConfigError("Invalid config"))

    with pytest.raises(ConfigError, match="Invalid config"):
        SupabaseDB(config)


def test_init_raises_supabase_connection_error(config):
    config.get_supabase_connection = Mock(side_effect=Exception("Connection failed"))

    with pytest.raises(
        SupabaseConnectionError,
        match="Connection failed",
    ):
        SupabaseDB(config)


def test_sign_in_success(mock_db):
    db, _, mock_auth = mock_db

    mock_user = Mock()
    mock_user.id = "123"
    mock_user.email = "test@example.com"

    mock_response = Mock()
    mock_response.user = mock_user

    mock_auth.sign_in_with_password.return_value = mock_response

    success, message = db.sign_in("test@example.com", "password123")

    assert success is True
    assert message == "Successfully logged in as test@example.com!"
    assert db.user_id == "123"
    assert db.user_email == "test@example.com"

    mock_auth.sign_in_with_password.assert_called_once_with(
        {
            "email": "test@example.com",
            "password": "password123",
        }
    )


def test_sign_in_no_user_returned(mock_db):
    db, _, mock_auth = mock_db

    mock_response = Mock()
    mock_response.user = None

    mock_auth.sign_in_with_password.return_value = mock_response

    success, message = db.sign_in(
        "test@example.com",
        "password123",
    )

    assert success is False
    assert message == "Login failed: No user returned."

    assert db.user_id is None
    assert db.user_email is None


def test_sign_in_exception(config, mock_db):
    db, _, mock_auth = mock_db
    mock_auth.sign_in_with_password.side_effect = Exception("Authentication failed")

    config.logger = Mock()

    success, message = db.sign_in(
        "test@example.com",
        "password123",
    )

    assert success is False
    assert message == "Login failed: Authentication failed"

    assert db.user_id is None
    assert db.user_email is None

    mock_auth.sign_in_with_password.assert_called_once_with(
        {
            "email": "test@example.com",
            "password": "password123",
        }
    )

    config.logger.error.assert_called_once()


def test_sign_up_success(mock_db):
    db, _, mock_auth = mock_db

    mock_user = Mock()
    mock_user.email = "test@example.com"

    mock_response = Mock()
    mock_response.user = mock_user

    mock_auth.sign_up.return_value = mock_response

    success, message = db.sign_up(
        "test@example.com",
        "password123",
    )

    assert success is True

    assert (
        message
        == "Registration successful! Please check your email (test@example.com) or log in."
    )

    mock_auth.sign_up.assert_called_once_with(
        {
            "email": "test@example.com",
            "password": "password123",
        }
    )


def test_sign_up_no_user_returned(mock_db):
    db, _, mock_auth = mock_db

    mock_response = Mock()
    mock_response.user = None

    mock_auth.sign_up.return_value = mock_response

    success, message = db.sign_up(
        "test@example.com",
        "password123",
    )

    assert success is False
    assert message == "Sign up failed: No user returned."

    mock_auth.sign_up.assert_called_once_with(
        {
            "email": "test@example.com",
            "password": "password123",
        }
    )


def test_sign_up_exception(config, mock_db):
    db, _, mock_auth = mock_db

    mock_auth.sign_up.side_effect = Exception("Registration failed")

    config.logger = Mock()

    success, message = db.sign_up(
        "test@example.com",
        "password123",
    )

    assert success is False
    assert message == "Sign up failed: Registration failed"

    mock_auth.sign_up.assert_called_once_with(
        {
            "email": "test@example.com",
            "password": "password123",
        }
    )

    config.logger.error.assert_called_once()


def test_sign_out_success(mock_db):
    db, _, mock_auth = mock_db

    db.user_id = "123"

    db.sign_out()

    assert db.user_id is None

    mock_auth.sign_out.assert_called_once()


def test_sign_out_exception_logs_warning(config, mock_db):
    db, _, mock_auth = mock_db

    db.user_id = "123"

    mock_auth.sign_out.side_effect = Exception("Logout failed")

    config.logger = Mock()

    db.sign_out()

    assert db.user_id == "123"

    mock_auth.sign_out.assert_called_once()

    config.logger.warning.assert_called_once()


def test_get_threads_success(mock_db):
    db, mock_supabase, _ = mock_db

    db.user_id = "123"

    mock_response = Mock()
    mock_response.data = [
        {
            "thread_id": "t1",
            "title": "Chat 1",
            "created_at": "2025-01-01",
        },
        {
            "thread_id": "t2",
            "title": "Chat 2",
            "created_at": "2025-01-02",
        },
    ]

    mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_response

    threads = db.get_threads()

    assert len(threads) == 2
    assert threads[0]["thread_id"] == "t1"
    assert threads[1]["thread_id"] == "t2"


def test_get_threads_no_user_returns_empty(mock_db):
    db, _, _ = mock_db

    db.user_id = None

    result = db.get_threads()

    assert result == []


def test_get_threads_exception_logs_error(mock_db, config):
    db, mock_supabase, _ = mock_db

    db.user_id = "123"

    mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.side_effect = Exception(
        "DB failure"
    )

    config.logger = Mock()

    result = db.get_threads()

    assert result == []

    config.logger.error.assert_called_once()


def test_load_chat_history_success(mock_db):
    db, mock_supabase, _ = mock_db

    db.user_id = "123"

    mock_response = Mock()
    mock_response.data = [
        {
            "role": "user",
            "type": "text",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "type": "text",
            "content": "Hi there",
        },
    ]

    mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

    result = db.load_chat_history("thread-1")

    assert len(result) == 2

    assert result[0]["role"] == "user"
    assert result[0]["type"] == "text"
    assert result[0]["content"] == "Hello"

    assert result[1]["role"] == "assistant"


def test_upsert_chat_thread_success(mock_db):
    db, mock_supabase, _ = mock_db

    db.user_id = "123"

    mock_table = Mock()
    mock_supabase.table.return_value = mock_table
    mock_table.upsert.return_value.execute.return_value = None

    db.upsert_chat_thread("thread-1", "My Title")

    mock_supabase.table.assert_called_once_with("threads")

    mock_table.upsert.assert_called_once_with(
        {
            "thread_id": "thread-1",
            "user_id": "123",
            "title": "My Title",
        }
    )

    mock_table.upsert.return_value.execute.assert_called_once()


def test_upsert_chat_thread_default_title(mock_db):
    db, mock_supabase, _ = mock_db

    db.user_id = "123"

    mock_table = Mock()
    mock_supabase.table.return_value = mock_table
    mock_table.upsert.return_value.execute.return_value = None

    db.upsert_chat_thread("thread-1")

    mock_table.upsert.assert_called_once_with(
        {
            "thread_id": "thread-1",
            "user_id": "123",
            "title": "New Conversation",
        }
    )


def test_upsert_chat_thread_no_user_returns_early(mock_db):
    db, mock_supabase, _ = mock_db

    db.user_id = None

    db.upsert_chat_thread("thread-1", "My Title")

    mock_supabase.table.assert_not_called()


def test_upsert_chat_thread_exception_logs_error(mock_db, config):
    db, mock_supabase, _ = mock_db

    db.user_id = "123"

    mock_table = Mock()
    mock_supabase.table.return_value = mock_table

    mock_table.upsert.return_value.execute.side_effect = Exception("DB failure")

    config.logger = Mock()

    db.upsert_chat_thread("thread-1", "My Title")

    config.logger.error.assert_called_once()


def test_load_chat_history_missing_type_defaults(mock_db):
    db, mock_supabase, _ = mock_db

    db.user_id = "123"

    mock_response = Mock()
    mock_response.data = [
        {
            "role": "user",
            "content": "Hello",
        }
    ]

    mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

    result = db.load_chat_history("thread-1")

    assert result[0]["type"] == "text"
    assert result[0]["content"] == "Hello"
    assert result[0]["role"] == "user"


def test_load_chat_history_no_user_returns_empty(mock_db):
    db, _, _ = mock_db

    db.user_id = None

    result = db.load_chat_history("thread-1")

    assert result == []


def test_load_chat_history_exception_returns_empty(mock_db, config):
    db, mock_supabase, _ = mock_db

    db.user_id = "123"

    mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.side_effect = Exception(
        "DB error"
    )

    config.logger = Mock()

    result = db.load_chat_history("thread-1")

    assert result == []

    config.logger.error.assert_called_once()


def test_save_message_success(mock_db):
    db, mock_supabase, _ = mock_db

    db.user_id = "123"

    mock_table = Mock()
    mock_supabase.table.return_value = mock_table
    mock_table.insert.return_value.execute.return_value = None

    result = db.save_message(
        thread_id="thread-1",
        role="user",
        content="Hello",
    )

    assert result is True

    mock_supabase.table.assert_called_once_with("messages")

    mock_table.insert.assert_called_once_with(
        {
            "thread_id": "thread-1",
            "user_id": "123",
            "role": "user",
            "type": "text",
            "content": "Hello",
        }
    )

    mock_table.insert.return_value.execute.assert_called_once()


def test_save_message_custom_type(mock_db):
    db, mock_supabase, _ = mock_db

    db.user_id = "123"

    mock_table = Mock()
    mock_supabase.table.return_value = mock_table
    mock_table.insert.return_value.execute.return_value = None

    result = db.save_message(
        thread_id="thread-1",
        role="assistant",
        content={"text": "Hi"},
        msg_type="json",
    )

    assert result is True

    mock_table.insert.assert_called_once_with(
        {
            "thread_id": "thread-1",
            "user_id": "123",
            "role": "assistant",
            "type": "json",
            "content": {"text": "Hi"},
        }
    )


def test_save_message_no_user_returns_false(mock_db):
    db, mock_supabase, _ = mock_db

    db.user_id = None

    result = db.save_message(
        thread_id="thread-1",
        role="user",
        content="Hello",
    )

    assert result is False

    mock_supabase.table.assert_not_called()


def test_save_message_exception_returns_false(mock_db, config):
    db, mock_supabase, _ = mock_db

    db.user_id = "123"

    mock_table = Mock()
    mock_supabase.table.return_value = mock_table

    mock_table.insert.return_value.execute.side_effect = Exception("DB failure")

    config.logger = Mock()

    result = db.save_message(
        thread_id="thread-1",
        role="user",
        content="Hello",
    )

    assert result is False

    config.logger.error.assert_called_once()


def test_delete_thread_success(mock_db):
    db, mock_supabase, _ = mock_db

    db._execute_query = Mock()

    mock_threads_table = Mock()
    mock_supabase.table.return_value = mock_threads_table
    mock_threads_table.delete.return_value.eq.return_value.execute.return_value = None

    result = db.delete_thread("thread-1")

    assert result is True

    assert db._execute_query.call_count == 3

    expected_tables = [
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
    ]

    db._execute_query.assert_has_calls(
        [
            call(
                f"DELETE FROM {t} WHERE thread_id = 'thread-1';",
                f"Cleaning up {t} for thread thread-1",
            )
            for t in expected_tables
        ],
        any_order=True,
    )

    mock_supabase.table.assert_called_once_with("threads")


def test_delete_thread_supabase_exception(mock_db, config):
    db, mock_supabase, _ = mock_db

    db._execute_query = Mock()

    mock_threads_table = Mock()
    mock_supabase.table.return_value = mock_threads_table

    mock_threads_table.delete.return_value.eq.return_value.execute.side_effect = (
        Exception("DB failure")
    )

    config.logger = Mock()

    result = db.delete_thread("thread-1")

    assert result is False

    config.logger.error.assert_called_once()


def test_delete_thread_execute_query_failure(mock_db, config):
    db, mock_supabase, _ = mock_db

    def failing_execute_query(*args, **kwargs):
        raise SupabaseConnectionError("Connection failed")

    db._execute_query = Mock(side_effect=failing_execute_query)

    config.logger = Mock()

    result = db.delete_thread("thread-1")

    assert result is False

    config.logger.error.assert_called()


def test_delete_user_success(mock_db):
    db, mock_supabase, _ = mock_db

    db.get_threads = Mock(
        return_value=[
            {"thread_id": "t1"},
            {"thread_id": "t2"},
        ]
    )

    db._execute_query = Mock()

    mock_supabase.auth.admin.delete_user.return_value = None

    result = db.delete_user("user-123")

    assert result is True

    assert db._execute_query.call_count == 3

    mock_supabase.auth.admin.delete_user.assert_called_once_with("user-123")


def test_delete_user_no_threads(mock_db):
    db, mock_supabase, _ = mock_db

    db.get_threads = Mock(return_value=[])

    db._execute_query = Mock()

    mock_supabase.auth.admin.delete_user.return_value = None

    result = db.delete_user("user-123")

    assert result is True

    db._execute_query.assert_not_called()

    mock_supabase.auth.admin.delete_user.assert_called_once_with("user-123")


def test_delete_user_execute_query_failure(mock_db, config):
    db, mock_supabase, _ = mock_db

    db.get_threads = Mock(return_value=[{"thread_id": "t1"}])

    def failing_query(*args, **kwargs):
        raise SupabaseConnectionError("DB connection failed")

    db._execute_query = Mock(side_effect=failing_query)

    config.logger = Mock()

    result = db.delete_user("user-123")

    assert result is False

    config.logger.error.assert_called()


def test_insert_schema_cache_dict_success(mock_db):
    db, mock_supabase, _ = mock_db

    db._execute_query = Mock()

    mock_table = Mock()
    mock_supabase.table.return_value = mock_table
    mock_table.insert.return_value.execute.return_value = None

    result = db.insert_schema_cache(
        "schema-key",
        {"a": 1, "b": 2},
    )

    assert result is True

    mock_table.insert.assert_called_once()


def test_insert_schema_cache_model_dump(mock_db):
    db, mock_supabase, _ = mock_db

    class FakeModel:
        def model_dump(self, mode=None):
            return {"x": 10}

    db._execute_query = Mock()

    mock_table = Mock()
    mock_supabase.table.return_value = mock_table
    mock_table.insert.return_value.execute.return_value = None

    result = db.insert_schema_cache(
        "schema-key",
        FakeModel(),
    )

    assert result is True

    mock_table.insert.assert_called_once()


def test_insert_schema_cache_retry_exhausted(mock_db, config):
    db, _, _ = mock_db

    config.logger = Mock()

    with patch(
        "src.database.db.retry_transient_network",
        side_effect=NetworkRetryExhaustedError("Retries failed"),
    ):
        result = db.insert_schema_cache("key", {"a": 1})

    assert result is False

    config.logger.error.assert_called_once()


def test_insert_schema_cache_invalid_content(mock_db, config):
    db, mock_supabase, _ = mock_db

    class BadObject:
        pass

    db._execute_query = Mock()

    mock_table = Mock()
    mock_supabase.table.return_value = mock_table
    mock_table.insert.return_value.execute.return_value = None

    result = db.insert_schema_cache("key", BadObject())

    assert result is True


def test_get_schema_cache_success(mock_db):
    db, mock_supabase, _ = mock_db

    mock_response = Mock()
    mock_response.data = [
        {
            "content": {"a": 1},
            "updated_at": "2025-01-01T00:00:00",
        }
    ]

    mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

    result = db.get_schema_cache("key")

    assert result["a"] == 1
    assert "updated_at" in result


def test_get_schema_cache_empty(mock_db):
    db, mock_supabase, _ = mock_db

    mock_response = Mock()
    mock_response.data = []

    mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

    result = db.get_schema_cache("key")

    assert result is None


def test_get_schema_cache_invalid_json(mock_db, config):
    db, mock_supabase, _ = mock_db

    mock_response = Mock()
    mock_response.data = [
        {
            "content": "invalid-json",
            "updated_at": "2025-01-01",
        }
    ]

    mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

    config.logger = Mock()

    result = db.get_schema_cache("key")

    assert result is None

    config.logger.warning.assert_called()


def test_get_schema_cache_retry_failure(mock_db, config):
    db, _, _ = mock_db

    config.logger = Mock()

    def failing_retry(fn):
        raise NetworkRetryExhaustedError("fail")

    with patch("src.database.db.retry_transient_network", failing_retry):
        result = db.get_schema_cache("key")

    assert result is None
    config.logger.warning.assert_called()


def test_create_tables_success(mock_db):
    db, _, _ = mock_db

    db._execute_query = Mock()

    db.create_tables()

    calls = [c.args[1] for c in db._execute_query.call_args_list]

    assert any("Creating threads table" in c for c in calls)
    assert any("Creating messages table" in c for c in calls)
    assert any("Creating schema_cache table" in c for c in calls)


def test_create_tables_labels_correct(mock_db):
    db, _, _ = mock_db

    captured = []

    def fake_execute(query, name):
        captured.append(name)

    db._execute_query = Mock(side_effect=fake_execute)

    db.create_tables()

    assert "Creating threads table" in captured
    assert "Creating messages table" in captured
    assert "Creating schema_cache table" in captured
    assert any("Adding policies" in c for c in captured)


def test_create_tables_execute_query_failure(mock_db):
    db, _, _ = mock_db

    def failing_execute(*args, **kwargs):
        raise SupabaseConnectionError("DB failure")

    db._execute_query = Mock(side_effect=failing_execute)

    try:
        db.create_tables()
        assert False
    except SupabaseConnectionError:
        assert True


def test_create_tables_generic_exception_wrapped(mock_db):
    db, _, _ = mock_db

    def failing_execute(*args, **kwargs):
        raise Exception("unexpected error")

    db._execute_query = Mock(side_effect=failing_execute)

    try:
        db.create_tables()
        assert False
    except SupabaseConnectionError:
        assert True


def test_drop_tables_success(mock_db):
    db, _, _ = mock_db

    db._execute_query = Mock()

    db.drop_tables()

    assert db._execute_query.call_count == 7

    expected_calls = [
        call("DROP TABLE IF EXISTS messages;", "Dropping messages table"),
        call("DROP TABLE IF EXISTS threads;", "Dropping threads table"),
        call("DROP TABLE IF EXISTS schema_cache;", "Dropping schema_cache table"),
        call(
            "DROP TABLE IF EXISTS checkpoint_writes;",
            "Dropping checkpoint_writes table",
        ),
        call(
            "DROP TABLE IF EXISTS checkpoint_blobs;", "Dropping checkpoint_blobs table"
        ),
        call(
            "DROP TABLE IF EXISTS checkpoint_migrations;",
            "Dropping checkpoint_migrations table",
        ),
        call("DROP TABLE IF EXISTS checkpoints;", "Dropping checkpoints table"),
    ]

    db._execute_query.assert_has_calls(expected_calls, any_order=False)


def test_drop_tables_ordering(mock_db):
    db, _, _ = mock_db

    calls = []

    def fake_execute(query, name):
        calls.append((query, name))

    db._execute_query = Mock(side_effect=fake_execute)

    db.drop_tables()

    assert calls[0][1] == "Dropping messages table"
    assert calls[-1][1] == "Dropping checkpoints table"


def test_drop_tables_execute_failure(mock_db):
    db, _, _ = mock_db

    def failing_execute(*args, **kwargs):
        raise SupabaseConnectionError("DB failure")

    db._execute_query = Mock(side_effect=failing_execute)

    try:
        db.drop_tables()
        assert False
    except SupabaseConnectionError:
        assert True


def test_drop_tables_generic_exception_wrapped(mock_db):
    db, _, _ = mock_db

    def failing_execute(*args, **kwargs):
        raise Exception("unexpected")

    db._execute_query = Mock(side_effect=failing_execute)

    try:
        db.drop_tables()
        assert False
    except SupabaseConnectionError:
        assert True
