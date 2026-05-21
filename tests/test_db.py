from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.config.config import Config
from src.database.db import SupabaseDB
from src.exceptions.exception import (
    ConfigError,
    NetworkRetryExhaustedError,
    SupabaseConnectionError,
)

# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture
def mock_config():
    """Provides a mocked Config template with mock logger and string properties."""
    config = MagicMock(spec=Config)
    config.logger = MagicMock()
    config.sb_db_uri = "postgresql://postgres:password@localhost:5432/postgres"
    return config


@pytest.fixture
def mock_supabase_client():
    """Provides a structural mock of the Supabase Client ecosystem."""
    client = MagicMock()
    client.auth = MagicMock()
    client.auth.admin = MagicMock()
    return client


@pytest.fixture
def db_instance(mock_config, mock_supabase_client):
    """Provides an initialized database engine targeting mock infrastructure."""
    mock_config.get_supabase_connection.return_value = mock_supabase_client
    return SupabaseDB(config=mock_config, admin=False)


# =====================================================================
# INITIALIZATION TESTS
# =====================================================================


def test_init_success(mock_config, mock_supabase_client):
    mock_config.get_supabase_connection.return_value = mock_supabase_client

    db = SupabaseDB(config=mock_config, admin=True)

    assert db.config == mock_config
    assert db.supabase_conn == mock_supabase_client
    assert db.user_id is None
    assert db.user_email is None
    mock_config.get_supabase_connection.assert_called_once_with(write_access=True)


def test_init_propagates_config_error(mock_config):
    mock_config.get_supabase_connection.side_effect = ConfigError("Invalid setup keys")

    with pytest.raises(ConfigError):
        SupabaseDB(config=mock_config)


def test_init_wraps_generic_exception(mock_config):
    mock_config.get_supabase_connection.side_effect = Exception("Sockets dropped")

    with pytest.raises(SupabaseConnectionError):
        SupabaseDB(config=mock_config)


# =====================================================================
# AUTHENTICATION TESTS
# =====================================================================


def test_sign_in_success(db_instance, mock_supabase_client):
    mock_user = MagicMock()
    mock_user.id = "uuid-111"
    mock_user.email = "test@user.com"

    mock_response = MagicMock()
    mock_response.user = mock_user
    mock_supabase_client.auth.sign_in_with_password.return_value = mock_response

    success, msg = db_instance.sign_in("test@user.com", "password")

    assert success is True
    assert "Successfully logged in" in msg
    assert db_instance.user_id == "uuid-111"
    assert db_instance.user_email == "test@user.com"
    mock_supabase_client.auth.sign_in_with_password.assert_called_once_with(
        {"email": "test@user.com", "password": "password"}
    )


def test_sign_in_failure_empty_response(db_instance, mock_supabase_client):
    mock_response = MagicMock()
    mock_response.user = None
    mock_supabase_client.auth.sign_in_with_password.return_value = mock_response

    success, msg = db_instance.sign_in("test@user.com", "password")

    assert success is False
    assert "Login failed: No user returned." in msg


def test_sign_in_exception_handling(db_instance, mock_supabase_client, mock_config):
    mock_supabase_client.auth.sign_in_with_password.side_effect = Exception(
        "Invalid signature"
    )

    success, msg = db_instance.sign_in("test@user.com", "password")

    assert success is False
    assert "Login failed:" in msg
    mock_config.logger.error.assert_called_once()


def test_sign_up_success(db_instance, mock_supabase_client):
    mock_user = MagicMock()
    mock_user.email = "new@user.com"
    mock_response = MagicMock()
    mock_response.user = mock_user
    mock_supabase_client.auth.sign_up.return_value = mock_response

    success, msg = db_instance.sign_up("new@user.com", "password")

    assert success is True
    assert "Registration successful!" in msg


def test_sign_up_failure_empty_response(db_instance, mock_supabase_client):
    mock_response = MagicMock()
    mock_response.user = None
    mock_supabase_client.auth.sign_up.return_value = mock_response

    success, msg = db_instance.sign_up("new@user.com", "password")

    assert success is False
    assert "Sign up failed: No user returned." in msg


def test_sign_up_exception_handling(db_instance, mock_supabase_client, mock_config):
    mock_supabase_client.auth.sign_up.side_effect = Exception("User already exists")

    success, msg = db_instance.sign_up("new@user.com", "password")

    assert success is False
    assert "Sign up failed:" in msg
    mock_config.logger.error.assert_called_once()


def test_sign_out_clears_local_state(db_instance, mock_supabase_client):
    db_instance.user_id = "uuid-111"

    db_instance.sign_out()

    assert db_instance.user_id is None
    mock_supabase_client.auth.sign_out.assert_called_once()


def test_sign_out_exception_logs_warning(
    db_instance, mock_supabase_client, mock_config
):
    mock_supabase_client.auth.sign_out.side_effect = Exception("Auth server down")

    # Should catch internally without crashing
    db_instance.sign_out()
    mock_config.logger.warning.assert_called_once()


# =====================================================================
# CHAT THREADS & HISTORY FLUENT QUERY TESTS
# =====================================================================


def test_get_threads_unauthenticated(db_instance):
    db_instance.user_id = None
    assert db_instance.get_threads() == []


def test_get_threads_success(db_instance, mock_supabase_client):
    db_instance.user_id = "uuid-111"
    expected_data = [{"thread_id": "t-1", "title": "Thread 1"}]

    mock_execute = mock_supabase_client.table.return_value.select.return_value.eq.return_value.order.return_value.execute
    mock_execute.return_value = MagicMock(data=expected_data)

    results = db_instance.get_threads()

    assert results == expected_data
    mock_supabase_client.table.assert_called_with("threads")


def test_get_threads_exception_returns_empty(
    db_instance, mock_supabase_client, mock_config
):
    db_instance.user_id = "uuid-111"
    mock_supabase_client.table.side_effect = Exception("DB Timeout")

    results = db_instance.get_threads()

    assert results == []
    mock_config.logger.error.assert_called_once()


def test_upsert_chat_thread_unauthenticated(db_instance, mock_supabase_client):
    db_instance.user_id = None
    db_instance.upsert_chat_thread("t-1")
    mock_supabase_client.table.assert_not_called()


def test_upsert_chat_thread_success(db_instance, mock_supabase_client):
    db_instance.user_id = "uuid-111"
    mock_upsert = mock_supabase_client.table.return_value.upsert

    db_instance.upsert_chat_thread(thread_id="t-1", title="My Title")

    mock_supabase_client.table.assert_called_with("threads")
    mock_upsert.assert_called_once_with(
        {"thread_id": "t-1", "user_id": "uuid-111", "title": "My Title"}
    )


def test_upsert_chat_thread_exception_logged(
    db_instance, mock_supabase_client, mock_config
):
    db_instance.user_id = "uuid-111"
    mock_supabase_client.table.side_effect = Exception("Schema locked")

    db_instance.upsert_chat_thread("t-1")
    mock_config.logger.error.assert_called_once()


def test_load_chat_history_unauthenticated(db_instance):
    db_instance.user_id = None
    assert db_instance.load_chat_history("t-1") == []


def test_load_chat_history_success_and_mapping(db_instance, mock_supabase_client):
    db_instance.user_id = "uuid-111"
    raw_payload = [
        {"role": "user", "type": "text", "content": "Hello", "id": 99},
        {
            "role": "assistant",
            "content": "Hi",
            "id": 100,
        },  # Tests missing type fallback
    ]

    mock_execute = mock_supabase_client.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute
    mock_execute.return_value = MagicMock(data=raw_payload)

    history = db_instance.load_chat_history("t-1")

    assert len(history) == 2
    assert history[0] == {"role": "user", "type": "text", "content": "Hello"}
    assert history[1] == {"role": "assistant", "type": "text", "content": "Hi"}


def test_load_chat_history_exception_returns_empty(
    db_instance, mock_supabase_client, mock_config
):
    db_instance.user_id = "uuid-111"
    mock_supabase_client.table.side_effect = Exception("Query timed out")

    assert db_instance.load_chat_history("t-1") == []
    mock_config.logger.error.assert_called_once()


def test_save_message_unauthenticated(db_instance):
    db_instance.user_id = None
    assert db_instance.save_message("t-1", "user", "hi") is False


def test_save_message_success(db_instance, mock_supabase_client):
    db_instance.user_id = "uuid-111"
    mock_insert = mock_supabase_client.table.return_value.insert

    result = db_instance.save_message("t-1", "user", "payload data", "text")

    assert result is True
    mock_insert.assert_called_once_with(
        {
            "thread_id": "t-1",
            "user_id": "uuid-111",
            "role": "user",
            "type": "text",
            "content": "payload data",
        }
    )


def test_save_message_exception_returns_false(
    db_instance, mock_supabase_client, mock_config
):
    db_instance.user_id = "uuid-111"
    mock_supabase_client.table.side_effect = Exception("Disk full")

    assert db_instance.save_message("t-1", "user", "hi") is False
    mock_config.logger.error.assert_called_once()


# =====================================================================
# DELETION OPERATIONS & RAW INTERPOLATION TESTS
# =====================================================================


@patch.object(SupabaseDB, "_execute_query")
def test_delete_thread_success(mock_execute_query, db_instance, mock_supabase_client):
    res = db_instance.delete_thread("thread-999")

    assert res is True
    # Verifies the 3 raw checkpoints queries executed
    assert mock_execute_query.call_count == 3
    # Verifies the cascades deleted the wrapper thread
    mock_supabase_client.table.assert_called_with("threads")


@patch.object(SupabaseDB, "_execute_query")
def test_delete_thread_handles_connection_error(
    mock_execute_query, db_instance, mock_config
):
    mock_execute_query.side_effect = SupabaseConnectionError("Cannot hit server")

    res = db_instance.delete_thread("thread-999")

    assert res is False
    mock_config.logger.error.assert_called_once()


@patch.object(SupabaseDB, "_execute_query")
def test_delete_thread_handles_generic_exception(
    mock_execute_query, db_instance, mock_config
):
    mock_execute_query.side_effect = Exception("Unexpected drop")

    res = db_instance.delete_thread("thread-999")

    assert res is False
    mock_config.logger.error.assert_called_once()


@patch.object(SupabaseDB, "_execute_query")
def test_delete_user_success_with_threads(
    mock_execute_query, db_instance, mock_supabase_client
):
    with patch.object(
        db_instance,
        "get_threads",
        return_value=[{"thread_id": "t1"}, {"thread_id": "t2"}],
    ):
        res = db_instance.delete_user("user-xyz")

    assert res is True
    mock_execute_query.assert_any_call(
        "DELETE FROM checkpoints WHERE thread_id IN ('t1', 't2');",
        "Cleaning up checkpoints",
    )
    mock_supabase_client.auth.admin.delete_user.assert_called_once_with("user-xyz")


def test_delete_user_success_no_threads(db_instance, mock_supabase_client):
    # Here we isolate the get_threads return value to an empty list
    with (
        patch.object(db_instance, "get_threads", return_value=[]),
        patch.object(db_instance, "_execute_query") as mock_execute_query,
    ):
        res = db_instance.delete_user("user-xyz")

        assert res is True
        mock_execute_query.assert_not_called()
        mock_supabase_client.auth.admin.delete_user.assert_called_once_with("user-xyz")


@patch.object(SupabaseDB, "get_threads")
def test_delete_user_admin_exception_handling(
    mock_get_threads, db_instance, mock_supabase_client, mock_config
):
    mock_get_threads.return_value = []
    mock_supabase_client.auth.admin.delete_user.side_effect = Exception(
        "Admin token expired"
    )

    res = db_instance.delete_user("user-xyz")

    assert res is False
    mock_config.logger.error.assert_called_once()


# =====================================================================
# SCHEMA CACHE (TRANSIENT NETWORK RETRY TESTING)
# =====================================================================


@patch("src.db.retry_transient_network")
def test_insert_schema_cache_serialization_error(
    mock_retry, db_instance, mock_supabase_client, mock_config
):
    # Force the underlying insert call to crash on serialization
    mock_supabase_client.table.return_value.insert.side_effect = TypeError(
        "Object of type set is not JSON serializable"
    )

    # Run the retry wrapper so it triggers the internal block
    mock_retry.side_effect = lambda func, *args, **kwargs: func()

    bad_content = {"invalid_json_field": {1, 2, 3}}
    res = db_instance.insert_schema_cache("cache_key", bad_content)

    assert res is False


@patch("src.database.db.retry_transient_network")
def test_insert_schema_cache_network_exhausted(mock_retry, db_instance, mock_config):
    mock_retry.side_effect = NetworkRetryExhaustedError(
        "Connection retries dropped out"
    )

    res = db_instance.insert_schema_cache("cache_key", {"data": 1})

    assert res is False
    mock_config.logger.error.assert_called_once()


@patch("src.database.db.retry_transient_network")
def test_get_schema_cache_returns_none_if_empty(mock_retry, db_instance):
    mock_response = MagicMock()
    mock_response.data = []
    mock_retry.return_value = mock_response

    assert db_instance.get_schema_cache("key") is None


@patch("src.database.db.retry_transient_network")
def test_get_schema_cache_success_string_json(mock_retry, db_instance):
    mock_response = MagicMock()
    mock_response.data = [
        {"content": '{"nodes": []}', "updated_at": datetime(2026, 5, 21, 12, 0)}
    ]
    mock_retry.return_value = mock_response

    payload = db_instance.get_schema_cache("key")
    assert payload["nodes"] == []
    assert payload["updated_at"] == "2026-05-21T12:00:00"


@patch("src.database.db.retry_transient_network")
def test_get_schema_cache_success_dict_json(mock_retry, db_instance):
    mock_response = MagicMock()
    mock_response.data = [
        {"content": {"nodes": [1]}, "updated_at": "2026-05-21T15:30:00"}
    ]
    mock_retry.return_value = mock_response

    payload = db_instance.get_schema_cache("key")
    assert payload["nodes"] == [1]
    assert payload["updated_at"] == "2026-05-21T15:30:00"


@patch("src.database.db.retry_transient_network")
def test_get_schema_cache_invalid_type_logs_warning(
    mock_retry, db_instance, mock_config
):
    mock_response = MagicMock()
    mock_response.data = [
        {"content": 12345, "updated_at": None}
    ]  # integers are unexpected content models
    mock_retry.return_value = mock_response

    assert db_instance.get_schema_cache("key") is None
    mock_config.logger.warning.assert_called_once()


@patch("src.database.db.retry_transient_network")
def test_get_schema_cache_malformed_json_string(mock_retry, db_instance, mock_config):
    mock_response = MagicMock()
    mock_response.data = [{"content": '{"unclosed_brace', "updated_at": None}]
    mock_retry.return_value = mock_response

    assert db_instance.get_schema_cache("key") is None
    assert mock_config.logger.warning.call_count == 1


# =====================================================================
# TABLE SCHEMA ORCHESTRATION & DDL EXECUTIONS
# =====================================================================


@patch.object(SupabaseDB, "_execute_query")
def test_create_tables_success(mock_execute, db_instance):
    db_instance.create_tables()
    # Confirms it loops sequentially through all migrations and policy configurations
    assert mock_execute.call_count == 14


@patch.object(SupabaseDB, "_execute_query")
def test_create_tables_exception_wrap(mock_execute, db_instance):
    mock_execute.side_effect = Exception("Syntax mismatch near POLICY")
    with pytest.raises(SupabaseConnectionError):
        db_instance.create_tables()


@patch.object(SupabaseDB, "_execute_query")
def test_drop_tables_success(mock_execute, db_instance):
    db_instance.drop_tables()
    assert mock_execute.call_count == 7


# =====================================================================
# PSYCOPG2 LOW LEVEL CONTEXT MANAGEMENT TESTING
# =====================================================================


@patch("src.database.db.psycopg2")
def test_execute_query_success_flow(mock_psycopg2, db_instance):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_psycopg2.connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    db_instance._execute_query("SELECT 1;", "test-action")

    mock_psycopg2.connect.assert_called_once_with(db_instance.config.sb_db_uri)
    mock_cursor.execute.assert_called_once_with("SELECT 1;")
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


@patch("src.database.db.psycopg2")
def test_execute_query_exception_performs_rollback(mock_psycopg2, db_instance):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = Exception("Relation does not exist")

    mock_psycopg2.connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with pytest.raises(SupabaseConnectionError):
        db_instance._execute_query("DROP UNKNOWN;", "error-action")

    mock_conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()
