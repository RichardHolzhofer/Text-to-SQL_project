from unittest.mock import Mock, mock_open, patch

import pytest

from src.exceptions.exception import MCPConfigError
from src.mcp.server import TextToSQLMCPServer


def test_mcp_server_initialization():
    with (
        patch("src.mcp.server.FastMCP") as mock_fastmcp,
        patch("src.mcp.server.Config") as mock_config,
    ):
        server = TextToSQLMCPServer()

        mock_fastmcp.assert_called_once_with("Text-to-SQL MCP")
        mock_config.assert_called_once_with("text-to-sql-mcp")

        assert server.mcp == mock_fastmcp.return_value
        assert server.config == mock_config.return_value


def test_mcp_tool_registration():
    mock_decorator = Mock(side_effect=lambda fn: fn)

    mock_mcp_instance = Mock()
    mock_mcp_instance.tool.return_value = mock_decorator

    with (
        patch("src.mcp.server.FastMCP", return_value=mock_mcp_instance),
        patch("src.mcp.server.Config"),
    ):
        TextToSQLMCPServer()

        mock_mcp_instance.tool.assert_called_once()
        assert mock_decorator.call_count == 1


def test_query_database_missing_credentials_raises_mcp_config_error(
    mcp_server,
):
    mcp_server.config.reload_mcp_credentials = Mock(return_value=(None, None, None))

    with pytest.raises(MCPConfigError) as exc_info:
        mcp_server.query_database("show me sales")

    assert "MCP_USER_EMAIL or MCP_USER_PASSWORD" in str(exc_info.value)


def test_query_database_uses_pinned_credentials(
    mcp_server,
    mock_langgraph_client,
):
    mock_get_client, mock_client = mock_langgraph_client

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=("user123", "test@test.com", None)
    )

    mock_client.runs.wait.return_value = {"answer": "Sales are good"}

    result = mcp_server.query_database("show me sales")

    assert "Sales are good" in result


def test_query_database_successful_login(
    mcp_server,
    mock_langgraph_client,
):
    mock_get_client, mock_client = mock_langgraph_client

    mock_db = Mock()
    mock_db.sign_in.return_value = (True, "success")
    mock_db.user_id = "user123"
    mock_db.user_email = "test@test.com"

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=(None, "test@test.com", "password")
    )

    with patch("src.mcp.server.SupabaseDB", return_value=mock_db):
        mock_client.runs.wait.return_value = {"answer": "Success"}

        result = mcp_server.query_database("sales")

    assert "Success" in result

    mock_db.sign_in.assert_called_once_with(
        "test@test.com",
        "password",
    )


def test_query_database_signup_fallback(
    mcp_server,
    mock_langgraph_client,
):
    mock_get_client, mock_client = mock_langgraph_client

    mock_db = Mock()

    mock_db.sign_in.side_effect = [
        (False, "login failed"),
        (True, "success"),
    ]

    mock_db.sign_up.return_value = (True, "registered")

    mock_db.user_id = "user123"
    mock_db.user_email = "test@test.com"

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=(None, "test@test.com", "password")
    )

    with patch("src.mcp.server.SupabaseDB", return_value=mock_db):
        mock_client.runs.wait.return_value = {"answer": "Success"}

        result = mcp_server.query_database("sales")

    assert "Success" in result

    assert mock_db.sign_in.call_count == 2
    mock_db.sign_up.assert_called_once()


def test_query_database_signup_failure_raises(
    mcp_server,
):
    mock_db = Mock()

    mock_db.sign_in.return_value = (
        False,
        "login failed",
    )

    mock_db.sign_up.return_value = (
        False,
        "signup failed",
    )

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=(None, "test@test.com", "password")
    )

    with patch("src.mcp.server.SupabaseDB", return_value=mock_db):
        with pytest.raises(MCPConfigError) as exc_info:
            mcp_server.query_database("sales")

    assert "Dynamic registration failed" in str(exc_info.value)


def test_query_database_email_confirmation_error(
    mcp_server,
):
    mock_db = Mock()

    mock_db.sign_in.side_effect = Exception("email not confirmed")

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=(None, "test@test.com", "password")
    )

    with patch("src.mcp.server.SupabaseDB", return_value=mock_db):
        with pytest.raises(MCPConfigError) as exc_info:
            mcp_server.query_database("sales")

    assert "Authentication Pending" in str(exc_info.value)


def test_query_database_thread_creation_failure(
    mcp_server,
    mock_langgraph_client,
):
    mock_get_client, mock_client = mock_langgraph_client

    mock_client.threads.create.side_effect = Exception("thread failure")

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=("user123", "test@test.com", None)
    )

    result = mcp_server.query_database("sales")

    assert "Pipeline Error:" in result
    assert "thread failure" in result


def test_query_database_run_failure(
    mcp_server,
    mock_langgraph_client,
):
    mock_get_client, mock_client = mock_langgraph_client

    mock_client.runs.wait.side_effect = Exception("graph failure")

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=("user123", "test@test.com", None)
    )

    result = mcp_server.query_database("sales")

    assert "Pipeline Error:" in result
    assert "graph failure" in result


def test_query_database_safety_warning(
    mcp_server,
    mock_langgraph_client,
):
    mock_get_client, mock_client = mock_langgraph_client

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=("user123", "test@test.com", None)
    )

    mock_client.runs.wait.return_value = {"answer": "flagged for safety reasons"}

    result = mcp_server.query_database("sales")

    assert "Safety Warning:" in result


def test_query_database_unsupported_query(
    mcp_server,
    mock_langgraph_client,
):
    mock_get_client, mock_client = mock_langgraph_client

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=("user123", "test@test.com", None)
    )

    mock_client.runs.wait.return_value = {
        "generated_sql": {"unsupported_explanation": "unsupported"}
    }

    result = mcp_server.query_database("sales")

    assert "Query Unsupported:" in result


def test_query_database_natural_language_answer(
    mcp_server,
    mock_langgraph_client,
):
    mock_get_client, mock_client = mock_langgraph_client

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=("user123", "test@test.com", None)
    )

    mock_client.runs.wait.return_value = {"answer": "Sales increased"}

    result = mcp_server.query_database("sales")

    assert "Sales increased" in result


def test_query_database_natural_language_with_fuzzy_warning(
    mcp_server,
    mock_langgraph_client,
):
    mock_get_client, mock_client = mock_langgraph_client

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=("user123", "test@test.com", None)
    )

    mock_client.runs.wait.return_value = {
        "answer": "Sales increased",
        "generated_sql": {"fuzzy_match_warning": "Approximate match used"},
    }

    result = mcp_server.query_database("sales")

    assert "Warning:" in result
    assert "Approximate match used" in result


def test_query_database_tabular_response(
    mcp_server,
    mock_langgraph_client,
):
    mock_get_client, mock_client = mock_langgraph_client

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=("user123", "test@test.com", None)
    )

    mock_client.runs.wait.return_value = {
        "router": {"route": "tab"},
        "tabular_answer": {
            "answer": "Here is the data",
            "data": [
                {"sales": 100},
                {"sales": 200},
            ],
        },
    }

    result = mcp_server.query_database("sales")

    assert "Here is the data" in result
    assert "|   sales |" in result


def test_query_database_empty_response_returns_unexpected_error(
    mcp_server,
    mock_langgraph_client,
):
    mock_get_client, mock_client = mock_langgraph_client

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=("user123", "test@test.com", None)
    )

    mock_client.runs.wait.return_value = {}

    result = mcp_server.query_database("sales")

    assert "Pipeline Error:" in result
    assert "unexpected or empty response structure" in result.lower()


def test_query_database_generates_csv_download_link(
    mcp_server,
    mock_langgraph_client,
):
    mock_get_client, mock_client = mock_langgraph_client

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=("user123", "test@test.com", None)
    )

    mock_client.runs.wait.return_value = {
        "answer": "Sales increased",
        "query_results": [{"sales": i} for i in range(6)],
    }

    mock_df = Mock()
    mock_df.to_csv.return_value = "csv,data"

    with (
        patch("src.mcp.server.os.makedirs"),
        patch("builtins.open", mock_open()),
        patch("src.mcp.server.pd.DataFrame", return_value=mock_df),
    ):
        result = mcp_server.query_database("sales")

    assert "downloads/" in result


def test_query_database_generates_host_csv_link(
    mcp_server,
    mock_langgraph_client,
):
    mock_get_client, mock_client = mock_langgraph_client

    mcp_server.config.reload_mcp_credentials = Mock(
        return_value=("user123", "test@test.com", None)
    )

    mock_client.runs.wait.return_value = {
        "answer": "Sales increased",
        "query_results": [{"sales": i} for i in range(6)],
    }

    mock_df = Mock()
    mock_df.to_csv.return_value = "csv,data"

    with (
        patch.dict(
            "os.environ",
            {"HOST_PROJECT_PATH": "/project"},
        ),
        patch("src.mcp.server.os.makedirs"),
        patch("builtins.open", mock_open()),
        patch("src.mcp.server.pd.DataFrame", return_value=mock_df),
    ):
        result = mcp_server.query_database("sales")

    assert "file://" in result
    assert "Open Full Results CSV" in result
