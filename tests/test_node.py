from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage

from src.exceptions.exception import NodeException, SchemaBuildError
from src.nodes.node import TextToSQLNodes
from src.states.state import (
    ConversationEvaluator,
    Router,
    SQLGenerator,
    Validator,
)


def test_init_success(config):
    mock_db_instance = Mock()

    with patch(
        "src.nodes.node.SupabaseDB", return_value=mock_db_instance
    ) as mock_db_class:
        node = TextToSQLNodes(config)

    mock_db_class.assert_called_once_with(config, admin=True)

    assert node.config == config
    assert node.smart_llm is not None
    assert node.fast_llm is not None

    assert node.max_retry == 3
    assert node.safety_limit == 1000

    assert isinstance(node.exclude_columns, list)
    assert "credit_card_amount" in node.exclude_columns


def test_init_failure_wrapped(config):
    with patch("src.nodes.node.SupabaseDB", side_effect=Exception("DB crash")):
        with pytest.raises(NodeException) as exc_info:
            TextToSQLNodes(config)

    assert "DB crash" in str(exc_info.value)


def test_input_guardrail_safe_path(node):
    state = Mock()
    state.question = "What are all customers?"
    state.chat_history = ["old message"]

    mock_guardrails = Mock()
    mock_guardrails.scan_user_input.return_value = "clean question"
    mock_guardrails.scan_history.return_value = ["clean history"]

    with patch("src.nodes.node.get_guardrails", return_value=mock_guardrails):
        result = node.input_guardrail_node(state, Mock())

    assert result["sanitized_question"] == "clean question"
    assert result["chat_history"] == ["clean history"]
    assert "answer" not in result


def test_input_guardrail_blocked_path(node):
    state = Mock()
    state.question = "DROP TABLE users;"
    state.chat_history = []

    mock_guardrails = Mock()
    mock_guardrails.scan_user_input.return_value = (
        "flagged for safety reasons: injection detected"
    )
    mock_guardrails.scan_history.return_value = []

    with patch("src.nodes.node.get_guardrails", return_value=mock_guardrails):
        result = node.input_guardrail_node(state, Mock())

    assert "answer" in result
    assert "flagged for safety reasons" in result["answer"]

    msg = result["chat_history"][0]
    assert isinstance(msg, AIMessage)
    assert "flagged" in msg.content


def test_input_guardrail_calls_guardrails(node):
    state = Mock()
    state.question = "hello"
    state.chat_history = []

    mock_guardrails = Mock()
    mock_guardrails.scan_user_input.return_value = "clean"
    mock_guardrails.scan_history.return_value = []

    with patch("src.nodes.node.get_guardrails", return_value=mock_guardrails):
        node.input_guardrail_node(state, Mock())

    mock_guardrails.scan_user_input.assert_called_once_with("hello")
    mock_guardrails.scan_history.assert_called_once_with(state.chat_history)


def test_conversation_evaluator_general_conversation(node, mock_run_prompt):
    state = Mock()
    state.sanitized_question = "Hello, how are you?"

    mock_response = Mock()
    mock_response.is_general_conversation = True

    mock_run_prompt.return_value = mock_response

    result = node.conversation_evaluator_node(state, Mock())

    assert result == {"is_general_conversation": True}

    mock_run_prompt.assert_called_once()


def test_conversation_evaluator_run_prompt_arguments(node, mock_run_prompt):
    state = Mock()
    state.sanitized_question = "What tables exist?"

    mock_response = Mock(spec=ConversationEvaluator)
    mock_response.is_general_conversation = False

    runnable_config = Mock()

    mock_run_prompt.return_value = mock_response
    node.conversation_evaluator_node(state, runnable_config)

    mock_run_prompt.assert_called_once_with(
        prompt_name="evaluate_conversation",
        variables={"question": "What tables exist?"},
        config=node.config,
        output_schema=ConversationEvaluator,
        use_fast_llm=True,
        runnable_config=runnable_config,
    )


def test_conversation_evaluator_exception_fallback(node, mock_run_prompt):
    state = Mock()
    state.sanitized_question = "Hi"

    mock_run_prompt.side_effect = Exception("LLM failure")

    result = node.conversation_evaluator_node(state, Mock())

    assert result == {"is_general_conversation": False}


def test_handle_general_conversation_success(node, mock_run_prompt):
    state = Mock()
    state.sanitized_question = "Hello"
    state.chat_history = ["previous message"]

    mock_response = Mock()
    mock_response.content = " Hello there! "

    mock_run_prompt.return_value = mock_response

    result = node.handle_general_conversation_node(state, Mock())

    assert result["answer"] == "Hello there!"

    assert len(result["chat_history"]) == 1
    assert isinstance(result["chat_history"][0], AIMessage)
    assert result["chat_history"][0].content == "Hello there!"


def test_handle_general_conversation_run_prompt_called_correctly(node, mock_run_prompt):
    state = Mock()
    state.sanitized_question = "Hi"
    state.chat_history = ["history"]

    runnable_config = Mock()

    mock_response = Mock()
    mock_response.content = "Hello!"

    mock_run_prompt.return_value = mock_response

    node.handle_general_conversation_node(state, runnable_config)

    mock_run_prompt.assert_called_once_with(
        prompt_name="general_conversation",
        variables={"question": "Hi"},
        config=node.config,
        chat_history=["history"],
        use_fast_llm=True,
        runnable_config=runnable_config,
    )


def test_handle_general_conversation_exception_fallback(node, mock_run_prompt):
    state = Mock()
    state.sanitized_question = "Hello"
    state.chat_history = []

    mock_run_prompt.side_effect = Exception("LLM failure")
    result = node.handle_general_conversation_node(state, Mock())

    expected = "I'm sorry, I'm having trouble responding right now."

    assert result["answer"] == expected

    assert isinstance(result["chat_history"][0], AIMessage)
    assert result["chat_history"][0].content == expected


def test_build_schema_uses_existing_schema(node):
    existing_schema = Mock()

    state = Mock()
    state.db_schema = existing_schema

    result = node.build_schema(state, Mock())

    assert result["db_schema"] == existing_schema

    assert isinstance(result["validator"], Validator)
    assert result["router"] is None
    assert result["generated_sql"] is None


def test_build_schema_uses_cache(
    node,
    mock_schema_validate,
):
    cached_schema = {"tables": []}

    validated_schema = Mock()

    node.db.get_schema_cache = Mock(return_value=cached_schema)

    mock_schema_validate.return_value = validated_schema

    state = Mock()
    state.db_schema = None
    state.force_refresh = False

    result = node.build_schema(state, Mock())

    node.db.get_schema_cache.assert_called_once_with("unified_schema")

    mock_schema_validate.assert_called_once_with(cached_schema)

    assert result["db_schema"] == validated_schema


def test_build_schema_force_refresh_skips_cache(
    node,
    mock_load_yaml,
    mock_dump_yaml,
    mock_run_prompt,
):
    state = Mock()
    state.db_schema = None
    state.force_refresh = True

    node.db.get_schema_cache = Mock()

    mock_load_yaml.side_effect = [
        {"models": []},
        {"models": []},
    ]

    relationship_response = Mock()
    relationship_response.relationships = []

    mock_run_prompt.return_value = relationship_response

    result = node.build_schema(state, Mock())

    node.db.get_schema_cache.assert_not_called()

    assert "db_schema" in result


def test_build_schema_full_rebuild(
    node, mock_load_yaml, mock_dump_yaml, mock_run_prompt, table_factory
):
    state = Mock()
    state.db_schema = None
    state.force_refresh = False

    node.db.get_schema_cache = Mock(return_value=None)
    node.db.insert_schema_cache = Mock()

    mart_yaml = {
        "models": [
            {
                "name": "customers",
                "columns": [{"name": "customer_id"}],
            }
        ]
    }

    enhancement_yaml = {"models": []}

    mock_load_yaml.side_effect = [mart_yaml, enhancement_yaml]

    mock_dump_yaml.return_value = "yaml_string"

    mock_table = table_factory(table_name="customers")

    mock_relationship_response = Mock()
    mock_relationship_response.relationships = []

    mock_run_prompt.side_effect = [
        mock_table,
        mock_relationship_response,
    ]

    state_result = node.build_schema(state, Mock())

    assert mock_load_yaml.call_count == 2

    node.db.insert_schema_cache.assert_called_once()

    assert "db_schema" in state_result


def test_build_schema_filters_sensitive_columns(
    node, mock_load_yaml, mock_dump_yaml, mock_run_prompt, table_factory
):
    state = Mock()
    state.db_schema = None
    state.force_refresh = True

    node.db.insert_schema_cache = Mock()

    mart_yaml = {
        "models": [
            {
                "name": "payments",
                "columns": [
                    {"name": "credit_card_amount"},
                    {"name": "safe_column"},
                ],
            }
        ]
    }

    enhancement_yaml = {"models": []}

    mock_load_yaml.side_effect = [mart_yaml, enhancement_yaml]

    mock_dump_yaml.return_value = "yaml"

    mock_table = table_factory(table_name="payments")

    relationship_response = Mock()
    relationship_response.relationships = []

    mock_run_prompt.side_effect = [
        mock_table,
        relationship_response,
    ]

    node.build_schema(state, Mock())

    dumped_model = mock_dump_yaml.call_args[0][0]

    columns = dumped_model["model"]["columns"]

    assert {"name": "safe_column"} in columns

    assert {"name": "credit_card_amount"} not in columns


def test_build_schema_wraps_exceptions(
    node,
    mock_load_yaml,
):
    state = Mock()
    state.db_schema = None
    state.force_refresh = True

    mock_load_yaml.side_effect = Exception("YAML crash")

    with pytest.raises(SchemaBuildError) as exc_info:
        node.build_schema(state, Mock())

    assert "YAML crash" in str(exc_info.value)


def test_route_format_success(node, mock_run_prompt):
    state = Mock()
    state.sanitized_question = "Show me total sales by month"

    mock_response = Router(route="tab")

    mock_run_prompt.return_value = mock_response

    result = node.route_format(state, Mock())

    assert result["router"] == mock_response


def test_route_format_prompt_called_correctly(node, mock_run_prompt):
    state = Mock()
    state.sanitized_question = "Hello"

    mock_response = Router(route="nl")
    mock_run_prompt.return_value = mock_response

    runnable_config = Mock()

    node.route_format(state, runnable_config)

    mock_run_prompt.assert_called_once_with(
        prompt_name="evaluate_intent",
        variables={"question": "Hello"},
        config=node.config,
        output_schema=Router,
        use_fast_llm=True,
        runnable_config=runnable_config,
    )


def test_route_format_fallback(node, mock_run_prompt):
    state = Mock()
    state.sanitized_question = "test"

    mock_run_prompt.side_effect = Exception("LLM failure")

    result = node.route_format(state, Mock())

    assert isinstance(result["router"], Router)
    assert result["router"].route == "nl"


def test_extract_semantic_concept_success(node, mock_run_prompt):
    state = Mock()
    state.sanitized_question = "What are the top selling products?"

    mock_llm_response = Mock()
    mock_llm_response.content = "top selling products"

    mock_run_prompt.return_value = mock_llm_response

    mock_embedding_model = Mock()
    mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]

    node.config.get_embedding_model = Mock(return_value=mock_embedding_model)

    result = node.extract_semantic_concept(state, Mock())

    assert isinstance(result["generated_sql"], SQLGenerator)

    assert result["generated_sql"].search_concept == "top selling products"
    assert result["generated_sql"].query_vector == [0.1, 0.2, 0.3]


def test_extract_semantic_concept_prompt_called(node, mock_run_prompt):
    state = Mock()
    state.sanitized_question = "Find good restaurants"

    mock_llm_response = Mock()
    mock_llm_response.content = "good restaurants"

    mock_run_prompt.return_value = mock_llm_response

    mock_embedding_model = Mock()
    mock_embedding_model.embed_query.return_value = [0.1]

    node.config.get_embedding_model = Mock(return_value=mock_embedding_model)

    runnable_config = Mock()

    node.extract_semantic_concept(state, runnable_config)

    mock_run_prompt.assert_called_once_with(
        prompt_name="extract_search_concept",
        variables={
            "question": "Find good restaurants",
            "target_language": node.review_language,
        },
        config=node.config,
        use_fast_llm=True,
        runnable_config=runnable_config,
    )


def test_extract_semantic_concept_embedding_failure(node, mock_run_prompt):
    state = Mock()
    state.sanitized_question = "test question"

    mock_llm_response = Mock()
    mock_llm_response.content = "concept"

    mock_run_prompt.return_value = mock_llm_response

    mock_embedding_model = Mock()
    mock_embedding_model.embed_query.side_effect = Exception("embedding failed")

    node.config.get_embedding_model = Mock(return_value=mock_embedding_model)

    with pytest.raises(NodeException) as exc_info:
        node.extract_semantic_concept(state, Mock())

    assert "embedding failed" in str(exc_info.value)


def test_semantic_query_generator_success(node, mock_run_prompt, sql_generator_factory):
    state = Mock()

    state.sanitized_question = "top products"
    state.chat_history = []
    state.db_schema = Mock()
    state.db_schema.model_dump.return_value = {"tables": []}

    state.generated_sql = None

    mock_run_prompt.return_value = sql_generator_factory(
        sql_query="SELECT * FROM products"
    )

    with patch(
        "src.nodes.node.AIMessage", side_effect=lambda content: {"msg": content}
    ):
        result = node.semantic_query_generator(state, Mock())

    assert result["generated_sql"].sql_query == "SELECT * FROM products"
    assert len(result["chat_history"]) == 1


def test_semantic_query_generator_prompt_called(
    node, mock_run_prompt, sql_generator_factory
):
    state = Mock()
    state.sanitized_question = "find sales"
    state.chat_history = []

    state.db_schema = Mock()
    state.db_schema.model_dump.return_value = {"tables": []}

    state.generated_sql = None

    mock_run_prompt.return_value = sql_generator_factory()

    runnable_config = Mock()

    node.semantic_query_generator(state, runnable_config)

    call_kwargs = mock_run_prompt.call_args.kwargs

    assert call_kwargs["prompt_name"] == "generate_semantic_review_sql"
    assert call_kwargs["variables"]["question"] == "find sales"
    assert call_kwargs["variables"]["threshold"] == node.semantic_search_threshold

    assert "tables" in call_kwargs["variables"]["schema_context"]


def test_semantic_query_generator_preserves_state(
    node, mock_run_prompt, sql_generator_factory
):
    state = Mock()

    state.sanitized_question = "sales"
    state.chat_history = []

    state.db_schema = Mock()
    state.db_schema.model_dump.return_value = {"tables": []}

    state.generated_sql = sql_generator_factory(
        sql_query="SELECT old",
        thought_process="old",
        search_concept="old concept",
        query_vector=[1, 2, 3],
    )

    mock_sql_output = sql_generator_factory(
        fuzzy_match_warning="some warning",
    )

    mock_run_prompt.return_value = mock_sql_output

    result = node.semantic_query_generator(state, Mock())

    updated = result["generated_sql"]

    assert updated.search_concept == "old concept"
    assert updated.query_vector == [1, 2, 3]


def test_semantic_query_generator_fuzzy_reset(
    node, mock_run_prompt, sql_generator_factory
):
    state = Mock()

    state.sanitized_question = "test"
    state.chat_history = []

    state.db_schema = Mock()
    state.db_schema.model_dump.return_value = {"tables": []}

    state.generated_sql = None

    mock_sql_output = sql_generator_factory(fuzzy_match_warning="some warning")

    mock_run_prompt.return_value = mock_sql_output

    result = node.semantic_query_generator(state, Mock())

    assert result["generated_sql"].fuzzy_match_warning == "some warning"


def test_semantic_query_generator_fuzzy_reset_clears_when_missing(
    node, mock_run_prompt, sql_generator_factory
):
    state = Mock()

    state.sanitized_question = "test"
    state.chat_history = []

    state.db_schema = Mock()
    state.db_schema.model_dump.return_value = {"tables": []}

    state.generated_sql = None

    mock_sql_output = sql_generator_factory()

    mock_run_prompt.return_value = mock_sql_output

    result = node.semantic_query_generator(state, Mock())

    assert result["generated_sql"].fuzzy_match_warning is None


def test_generate_sql_happy_path(node, mock_run_prompt, sql_generator_factory):
    state = Mock()
    state.sanitized_question = "list customers"
    state.chat_history = []
    state.db_schema = Mock()
    state.db_schema.model_dump.return_value = {"tables": []}
    state.validator = Mock(iteration_count=0)

    mock_output = sql_generator_factory(
        sql_query="SELECT * FROM customers",
        thought_process="step by step",
        fuzzy_match_warning="warning",
    )

    mock_run_prompt.return_value = mock_output

    result = node.generate_sql(state, Mock())

    assert result["generated_sql"].sql_query == "SELECT * FROM customers"
    assert len(result["chat_history"]) == 1


def test_generate_sql_unsupported_path(node, mock_run_prompt, sql_generator_factory):
    state = Mock()
    state.sanitized_question = "hello"
    state.chat_history = []
    state.db_schema = Mock()
    state.db_schema.model_dump.return_value = {"tables": []}
    state.validator = Mock(iteration_count=0)

    mock_output = sql_generator_factory(
        sql_query=None,
        thought_process="",
        unsupported_explanation="I cannot answer this",
    )

    mock_run_prompt.return_value = mock_output

    result = node.generate_sql(state, Mock())

    assert result["chat_history"][0].content == "I cannot answer this"


def test_generate_sql_fuzzy_reset(node, mock_run_prompt, sql_generator_factory):
    state = Mock()
    state.sanitized_question = "test"
    state.chat_history = []
    state.db_schema = Mock()
    state.db_schema.model_dump.return_value = {"tables": []}
    state.validator = Mock(iteration_count=0)

    mock_output = sql_generator_factory(fuzzy_match_warning="old warning")

    mock_run_prompt.return_value = mock_output

    result = node.generate_sql(state, Mock())

    assert result["generated_sql"].fuzzy_match_warning == "old warning"


def test_validate_sql_success(node, mock_snowflake_connection, validator_factory):
    conn, cursor = mock_snowflake_connection

    node.config.get_snowflake_connection = Mock(return_value=conn)

    state = Mock()
    state.generated_sql = Mock(sql_query="SELECT 1", query_vector=None)
    state.validator = validator_factory()

    cursor.fetchmany.return_value = [(1,)]
    cursor.description = [("col",)]

    result = node.validate_sql(state, Mock())

    cursor.execute.assert_called_once()
    assert result["validator"].is_valid_query is True


def test_validate_sql_no_sql_generated(node, validator_factory):
    state = Mock()

    state.generated_sql = None
    state.validator = validator_factory(
        is_valid_query=True,
        error_message=None,
        iteration_count=0,
    )

    result = node.validate_sql(state, Mock())

    assert result["validator"].is_valid_query is False

    assert result["validator"].iteration_count == 1

    assert "No SQL generated to validate." in result["validator"].error_message


def test_validate_sql_injects_query_vector(node, mock_snowflake_connection):
    conn, cursor = mock_snowflake_connection

    node.config.get_snowflake_connection = Mock(return_value=conn)

    state = Mock()
    state.generated_sql = Mock(
        sql_query="SELECT [QUERY_VECTOR]", query_vector=[1, 2, 3]
    )
    state.validator = Mock(iteration_count=0)

    node.validate_sql(state, Mock())

    executed_sql = cursor.execute.call_args[0][0]

    assert "[QUERY_VECTOR]" not in executed_sql
    assert "1" in executed_sql


def test_validate_sql_explain_failure(
    node, mock_snowflake_connection, validator_factory
):
    conn, cursor = mock_snowflake_connection

    cursor.execute.side_effect = Exception("bad SQL")

    node.config.get_snowflake_connection = Mock(return_value=conn)

    state = Mock()
    state.generated_sql = Mock(sql_query="SELECT 1", query_vector=None)
    state.validator = validator_factory()

    result = node.validate_sql(state, Mock())

    assert result["validator"].is_valid_query is False
    assert "bad SQL" in result["validator"].error_message


def test_execute_sql_success(
    node,
    mock_snowflake_connection,
    validator_factory,
):
    conn, cursor = mock_snowflake_connection

    node.config.get_snowflake_connection = Mock(return_value=conn)

    state = Mock()

    state.validator = validator_factory(
        is_valid_query=True,
        error_message=None,
        iteration_count=0,
    )

    state.generated_sql = Mock(
        sql_query="SELECT 1",
        query_vector=None,
    )

    cursor.fetchmany.return_value = [(1,)]
    cursor.description = [("col",)]

    result = node.execute_sql(state, Mock())

    cursor.execute.assert_called_once_with("SELECT 1")

    assert result["query_results"] == [{"col": 1}]

    assert result["is_capped"] is False


def test_execute_sql_skips_invalid_query(
    node,
    mock_snowflake_connection,
    validator_factory,
):
    conn, cursor = mock_snowflake_connection

    node.config.get_snowflake_connection = Mock(return_value=conn)

    state = Mock()

    state.validator = validator_factory(
        is_valid_query=False,
    )

    state.generated_sql = Mock(
        sql_query="SELECT 1",
    )

    result = node.execute_sql(state, Mock())

    assert result == {"query_results": None}

    cursor.execute.assert_not_called()


def test_execute_sql_skips_missing_generated_sql(
    node,
    mock_snowflake_connection,
    validator_factory,
):
    conn, cursor = mock_snowflake_connection

    node.config.get_snowflake_connection = Mock(return_value=conn)

    state = Mock()

    state.validator = validator_factory(
        is_valid_query=True,
    )

    state.generated_sql = None

    result = node.execute_sql(state, Mock())

    assert result == {"query_results": None}

    cursor.execute.assert_not_called()


def test_execute_sql_skips_empty_sql_query(
    node,
    mock_snowflake_connection,
    validator_factory,
):
    conn, cursor = mock_snowflake_connection

    node.config.get_snowflake_connection = Mock(return_value=conn)

    state = Mock()

    state.validator = validator_factory(
        is_valid_query=True,
    )

    state.generated_sql = Mock(
        sql_query="",
    )

    result = node.execute_sql(state, Mock())

    assert result == {"query_results": None}

    cursor.execute.assert_not_called()


def test_execute_sql_empty_results(
    node,
    mock_snowflake_connection,
    validator_factory,
):
    conn, cursor = mock_snowflake_connection

    node.config.get_snowflake_connection = Mock(return_value=conn)

    state = Mock()

    state.validator = validator_factory(
        is_valid_query=True,
        iteration_count=0,
    )

    state.generated_sql = Mock(
        sql_query="SELECT 1",
        query_vector=None,
    )

    cursor.fetchmany.return_value = []
    cursor.description = [("col",)]

    result = node.execute_sql(state, Mock())

    cursor.execute.assert_called_once_with("SELECT 1")

    assert result["validator"].is_valid_query is False
    assert result["validator"].iteration_count == 1

    assert (
        result["validator"].error_message
        == "Query executed successfully but returned 0 results."
    )


def test_execute_sql_execution_exception(
    node,
    mock_snowflake_connection,
    validator_factory,
):
    conn, cursor = mock_snowflake_connection

    node.config.get_snowflake_connection = Mock(return_value=conn)

    cursor.execute.side_effect = Exception("Snowflake execution failed")

    state = Mock()

    state.validator = validator_factory(
        is_valid_query=True,
        iteration_count=0,
    )

    state.generated_sql = Mock(
        sql_query="SELECT 1",
        query_vector=None,
    )

    result = node.execute_sql(state, Mock())

    assert result["validator"].is_valid_query is False

    assert result["validator"].iteration_count == 1

    assert result["validator"].error_message == "Snowflake execution failed"


def test_execute_sql_query_vector_success(
    node,
    mock_snowflake_connection,
    validator_factory,
):
    conn, cursor = mock_snowflake_connection

    node.config.get_snowflake_connection = Mock(return_value=conn)

    state = Mock()

    state.validator = validator_factory(
        is_valid_query=True,
    )

    state.generated_sql = Mock(
        sql_query="SELECT * FROM table WHERE embedding = [QUERY_VECTOR]",
        query_vector=[0.1, 0.2, 0.3],
    )

    cursor.fetchmany.return_value = [(1,)]
    cursor.description = [("col",)]

    result = node.execute_sql(state, Mock())

    cursor.execute.assert_called_once_with(
        "SELECT * FROM table WHERE embedding = [0.1, 0.2, 0.3]"
    )

    assert result["query_results"] == [{"col": 1}]


def test_execute_sql_query_vector_missing(
    node,
    mock_snowflake_connection,
    validator_factory,
):
    conn, cursor = mock_snowflake_connection

    node.config.get_snowflake_connection = Mock(return_value=conn)

    state = Mock()

    state.validator = validator_factory(
        is_valid_query=True,
        iteration_count=0,
    )

    state.generated_sql = Mock(
        sql_query="SELECT * FROM table WHERE embedding = [QUERY_VECTOR]",
        query_vector=None,
    )

    result = node.execute_sql(state, Mock())

    assert result["validator"].is_valid_query is False

    assert result["validator"].iteration_count == 1

    assert "query_vector is missing" in result["validator"].error_message

    cursor.execute.assert_not_called()


def test_execute_sql_capped_results(
    node,
    mock_snowflake_connection,
    validator_factory,
):
    conn, cursor = mock_snowflake_connection

    node.config.get_snowflake_connection = Mock(return_value=conn)

    node.safety_limit = 2

    state = Mock()

    state.validator = validator_factory(
        is_valid_query=True,
    )

    state.generated_sql = Mock(
        sql_query="SELECT 1",
        query_vector=None,
    )

    cursor.fetchmany.return_value = [
        (1,),
        (2,),
        (3,),
    ]

    cursor.description = [("col",)]

    result = node.execute_sql(state, Mock())

    assert result["is_capped"] is True

    assert result["query_results"] == [
        {"col": 1},
        {"col": 2},
    ]


def test_results_guardrail_node_success(node):
    state = Mock()

    state.query_results = [
        {
            "customer_id": "12345",
            "name": "John",
        }
    ]

    mock_guardrails = Mock()

    mock_guardrails.scan_data.return_value = [
        {
            "customer_id": "[REDACTED]",
            "name": "John",
        }
    ]

    with patch(
        "src.nodes.node.get_guardrails",
        return_value=mock_guardrails,
    ):
        result = node.results_guardrail_node(state, Mock())

    assert result == {
        "sanitized_query_results": [
            {
                "customer_id": "[REDACTED]",
                "name": "John",
            }
        ]
    }

    mock_guardrails.scan_data.assert_called_once_with(
        [
            {
                "customer_id": "12345",
                "name": "John",
            }
        ]
    )


def test_results_guardrail_node_no_results(node):
    state = Mock()

    state.query_results = None

    with patch("src.nodes.node.get_guardrails") as mock_get_guardrails:
        result = node.results_guardrail_node(state, Mock())

    assert result == {"sanitized_query_results": None}

    mock_get_guardrails.assert_not_called()


def test_results_guardrail_node_empty_results(node):
    state = Mock()

    state.query_results = []

    with patch("src.nodes.node.get_guardrails") as mock_get_guardrails:
        result = node.results_guardrail_node(state, Mock())

    assert result == {"sanitized_query_results": None}

    mock_get_guardrails.assert_not_called()


def test_results_guardrail_node_scan_data_called_once(node):
    state = Mock()

    state.query_results = [{"id": "1"}]

    mock_guardrails = Mock()

    mock_guardrails.scan_data.return_value = [{"id": "[REDACTED]"}]

    with patch(
        "src.nodes.node.get_guardrails",
        return_value=mock_guardrails,
    ):
        node.results_guardrail_node(state, Mock())

    mock_guardrails.scan_data.assert_called_once()


def test_results_guardrail_node_guardrail_exception(node):
    state = Mock()

    state.query_results = [{"id": "1"}]

    mock_guardrails = Mock()

    mock_guardrails.scan_data.side_effect = Exception("Guardrail failure")

    with patch(
        "src.nodes.node.get_guardrails",
        return_value=mock_guardrails,
    ):
        try:
            node.results_guardrail_node(state, Mock())
            assert False
        except Exception as e:
            assert str(e) == "Guardrail failure"


def test_deanonymize_sql_node_success(node):
    state = Mock()

    generated_sql = Mock()

    generated_sql.sql_query = (
        "SELECT * FROM customers WHERE customer_id = '[ANONYMIZED]'"
    )

    restored_sql = "SELECT * FROM customers WHERE customer_id = '12345'"

    updated_generated_sql = Mock()
    updated_generated_sql.sql_query = restored_sql

    generated_sql.model_copy.return_value = updated_generated_sql

    state.generated_sql = generated_sql

    state.sanitized_question = "Show purchases for customer 12345"

    mock_guardrails = Mock()

    mock_guardrails.scan_llm_output.return_value = restored_sql

    with patch(
        "src.nodes.node.get_guardrails",
        return_value=mock_guardrails,
    ):
        result = node.deanonymize_sql_node(state, Mock())

    assert result["generated_sql"] == updated_generated_sql

    mock_guardrails.scan_llm_output.assert_called_once_with(
        "Show purchases for customer 12345",
        "SELECT * FROM customers WHERE customer_id = '[ANONYMIZED]'",
    )

    generated_sql.model_copy.assert_called_once_with(update={"sql_query": restored_sql})


def test_deanonymize_sql_node_no_generated_sql(node):
    state = Mock()

    state.generated_sql = None

    result = node.deanonymize_sql_node(state, Mock())

    assert result == {}


def test_deanonymize_sql_node_empty_sql_query(node):
    state = Mock()

    generated_sql = Mock()
    generated_sql.sql_query = ""

    state.generated_sql = generated_sql

    result = node.deanonymize_sql_node(state, Mock())

    assert result == {}


def test_deanonymize_sql_node_scan_llm_output_called_once(node):
    state = Mock()

    generated_sql = Mock()

    generated_sql.sql_query = "SELECT * FROM customers"

    generated_sql.model_copy.return_value = generated_sql

    state.generated_sql = generated_sql

    state.sanitized_question = "Show all customers"

    mock_guardrails = Mock()

    mock_guardrails.scan_llm_output.return_value = "SELECT * FROM customers"

    with patch(
        "src.nodes.node.get_guardrails",
        return_value=mock_guardrails,
    ):
        node.deanonymize_sql_node(state, Mock())

    mock_guardrails.scan_llm_output.assert_called_once()


def test_deanonymize_sql_node_guardrail_exception(node):
    state = Mock()

    generated_sql = Mock()

    generated_sql.sql_query = "SELECT * FROM customers"

    state.generated_sql = generated_sql

    state.sanitized_question = "Show all customers"

    mock_guardrails = Mock()

    mock_guardrails.scan_llm_output.side_effect = Exception("Deanonymization failure")

    with patch(
        "src.nodes.node.get_guardrails",
        return_value=mock_guardrails,
    ):
        try:
            node.deanonymize_sql_node(state, Mock())
            assert False
        except Exception as e:
            assert str(e) == "Deanonymization failure"


def test_deanonymize_answer_node_success(node):
    state = Mock()

    state.answer = "Customer [ANONYMIZED] placed 5 orders."

    state.sanitized_question = "How many orders did customer 12345 place?"

    restored_answer = "Customer 12345 placed 5 orders."

    mock_guardrails = Mock()

    mock_guardrails.scan_llm_output.return_value = restored_answer

    with patch(
        "src.nodes.node.get_guardrails",
        return_value=mock_guardrails,
    ):
        result = node.deanonymize_answer_node(state, Mock())

    assert result == {"answer": restored_answer}

    mock_guardrails.scan_llm_output.assert_called_once_with(
        "How many orders did customer 12345 place?",
        "Customer [ANONYMIZED] placed 5 orders.",
    )


def test_deanonymize_answer_node_no_answer(node):
    state = Mock()

    state.answer = None

    result = node.deanonymize_answer_node(state, Mock())

    assert result == {}


def test_deanonymize_answer_node_empty_answer(node):
    state = Mock()

    state.answer = ""

    result = node.deanonymize_answer_node(state, Mock())

    assert result == {}


def test_deanonymize_answer_node_scan_llm_output_called_once(node):
    state = Mock()

    state.answer = "Customer [ANONYMIZED] placed 5 orders."

    state.sanitized_question = "How many orders did customer 12345 place?"

    mock_guardrails = Mock()

    mock_guardrails.scan_llm_output.return_value = "Customer 12345 placed 5 orders."

    with patch(
        "src.nodes.node.get_guardrails",
        return_value=mock_guardrails,
    ):
        node.deanonymize_answer_node(state, Mock())

    mock_guardrails.scan_llm_output.assert_called_once()


def test_deanonymize_answer_node_guardrail_exception(node):
    state = Mock()

    state.answer = "Customer [ANONYMIZED] placed 5 orders."

    state.sanitized_question = "How many orders did customer 12345 place?"

    mock_guardrails = Mock()

    mock_guardrails.scan_llm_output.side_effect = Exception("Deanonymization failure")

    with patch(
        "src.nodes.node.get_guardrails",
        return_value=mock_guardrails,
    ):
        try:
            node.deanonymize_answer_node(state, Mock())
            assert False
        except Exception as e:
            assert str(e) == "Deanonymization failure"


def test_generate_tabular_answer_standard_success(node):
    state = Mock()

    state.router = Mock(is_semantic_intent=False)

    state.query_results = [
        {"id": 1},
        {"id": 2},
    ]

    state.is_capped = False

    result = node.generate_tabular_answer(state, Mock())

    tabular_answer = result["tabular_answer"]

    assert tabular_answer.data == [
        {"id": 1},
        {"id": 2},
    ]

    assert tabular_answer.answer == "Here are the tabular results you requested."

    assert tabular_answer.total_count == 2

    assert tabular_answer.is_capped is False

    assert (
        result["chat_history"][0].content
        == "Here are the tabular results you requested."
    )


def test_generate_tabular_answer_truncated_results(node):
    node.standard_tab_limit = 2

    state = Mock()

    state.router = Mock(is_semantic_intent=False)

    state.query_results = [
        {"id": 1},
        {"id": 2},
        {"id": 3},
    ]

    state.is_capped = False

    result = node.generate_tabular_answer(state, Mock())

    tabular_answer = result["tabular_answer"]

    assert tabular_answer.data == [
        {"id": 1},
        {"id": 2},
    ]

    assert (
        tabular_answer.answer
        == "Showing the first 2 records out of 3 total matches found. The full dataset can be downloaded as a CSV below."
    )

    assert tabular_answer.total_count == 3

    assert tabular_answer.is_capped is False


def test_generate_tabular_answer_semantic_limit(node):
    node.semantic_tab_limit = 1

    state = Mock()

    state.router = Mock(is_semantic_intent=True)

    state.query_results = [
        {"id": 1},
        {"id": 2},
    ]

    state.is_capped = False

    result = node.generate_tabular_answer(state, Mock())

    tabular_answer = result["tabular_answer"]

    assert tabular_answer.data == [
        {"id": 1},
    ]

    assert (
        tabular_answer.answer
        == "Showing the first 1 records out of 2 total matches found. The full dataset can be downloaded as a CSV below."
    )


def test_generate_tabular_answer_capped_results(node):
    node.standard_tab_limit = 2
    node.safety_limit = 1000

    state = Mock()

    state.router = Mock(is_semantic_intent=False)

    state.query_results = [
        {"id": 1},
        {"id": 2},
        {"id": 3},
    ]

    state.is_capped = True

    state.generated_sql = Mock(sql_query="SELECT * FROM customers")

    result = node.generate_tabular_answer(state, Mock())

    tabular_answer = result["tabular_answer"]

    assert tabular_answer.is_capped is True

    assert "safety limit of 1000 records" in tabular_answer.answer

    assert "SELECT * FROM customers" in tabular_answer.answer


def test_generate_tabular_answer_capped_results_without_generated_sql(node):
    node.standard_tab_limit = 2

    state = Mock()

    state.router = Mock(is_semantic_intent=False)

    state.query_results = [
        {"id": 1},
        {"id": 2},
        {"id": 3},
    ]

    state.is_capped = True

    state.generated_sql = None

    result = node.generate_tabular_answer(state, Mock())

    tabular_answer = result["tabular_answer"]

    assert "```sql\nN/A\n```" in tabular_answer.answer


def test_generate_tabular_answer_empty_results(node):
    state = Mock()

    state.router = Mock(is_semantic_intent=False)

    state.query_results = []

    state.is_capped = False

    result = node.generate_tabular_answer(state, Mock())

    tabular_answer = result["tabular_answer"]

    assert tabular_answer.data == []

    assert tabular_answer.answer == "Here are the tabular results you requested."

    assert tabular_answer.total_count == 0

    assert tabular_answer.is_capped is False


def test_generate_tabular_answer_none_results(node):
    state = Mock()

    state.router = Mock(is_semantic_intent=False)

    state.query_results = None

    state.is_capped = False

    result = node.generate_tabular_answer(state, Mock())

    tabular_answer = result["tabular_answer"]

    assert tabular_answer.data == []

    assert tabular_answer.total_count == 0

    assert tabular_answer.is_capped is False


def test_generate_tabular_answer_chat_history_contains_answer(node):
    state = Mock()

    state.router = Mock(is_semantic_intent=False)

    state.query_results = [
        {"id": 1},
    ]

    state.is_capped = False

    result = node.generate_tabular_answer(state, Mock())

    assert len(result["chat_history"]) == 1

    assert result["chat_history"][0].content == result["tabular_answer"].answer


def test_generate_nl_answer_success(node, mock_run_prompt):
    state = Mock()

    state.sanitized_question = "What are customers?"
    state.chat_history = []
    state.sanitized_query_results = [{"id": 1}, {"id": 2}]
    state.is_capped = False
    state.generated_sql = None

    mock_response = Mock()
    mock_response.content = "These are customers."
    mock_run_prompt.return_value = mock_response

    result = node.generate_nl_answer(state, Mock())

    assert result["answer"] == "These are customers."
    assert result["chat_history"][0].content == "These are customers."

    mock_run_prompt.assert_called_once()


def test_generate_nl_answer_unsupported_sql(node, mock_run_prompt):
    state = Mock()

    state.sanitized_question = "test"
    state.chat_history = []
    state.sanitized_query_results = []
    state.is_capped = False

    state.generated_sql = Mock(unsupported_explanation="I cannot answer this.")

    result = node.generate_nl_answer(state, Mock())

    assert result["answer"] == "I cannot answer this."
    assert result["chat_history"][0].content == "I cannot answer this."

    mock_run_prompt.assert_not_called()


def test_generate_nl_answer_capped(
    node,
    mock_run_prompt,
    sql_generator_factory,
):
    state = Mock()

    state.sanitized_question = "test"
    state.chat_history = []

    state.sanitized_query_results = [{"id": 1}]
    state.is_capped = True

    state.generated_sql = sql_generator_factory(
        sql_query="SELECT * FROM table",
        unsupported_explanation=None,
    )

    mock_run_prompt.return_value = SimpleNamespace(content="Answer")

    result = node.generate_nl_answer(state, Mock())

    assert "safety limit" in result["answer"]
    assert "SELECT * FROM table" in result["answer"]


def test_generate_nl_answer_no_results(node, mock_run_prompt):
    state = Mock()

    state.sanitized_question = "test"
    state.chat_history = []
    state.sanitized_query_results = None
    state.is_capped = False
    state.generated_sql = None

    mock_response = Mock()
    mock_response.content = "No results found."
    mock_run_prompt.return_value = mock_response

    mock_run_prompt.return_value = mock_response

    result = node.generate_nl_answer(state, Mock())

    assert result["answer"] == "No results found."


def test_generate_nl_answer_exception(node, mock_run_prompt):
    state = Mock()

    state.sanitized_question = "test"
    state.chat_history = []
    state.sanitized_query_results = [{"id": 1}]
    state.is_capped = False
    state.generated_sql = None

    mock_run_prompt.side_effect = Exception("LLM failed")

    try:
        node.generate_nl_answer(state, Mock())
        assert False
    except Exception as e:
        assert "Failed to generate NL answer" in str(e)


def test_generate_nl_answer_truncates_large_input(node, mock_run_prompt):
    state = Mock()

    state.sanitized_question = "test"
    state.chat_history = []

    state.sanitized_query_results = [{"id": "x" * 1000}] * 200
    state.is_capped = False
    state.generated_sql = None

    mock_response = Mock()
    mock_response.content = "OK"
    mock_run_prompt.return_value = mock_response

    result = node.generate_nl_answer(state, Mock())

    assert "answer" in result
    assert result["answer"] == "OK"


def test_summarize_review_sentiment_success(
    node,
    mock_run_prompt,
    sql_generator_factory,
):
    state = Mock()

    state.sanitized_question = "Summarize reviews"
    state.chat_history = []

    state.sanitized_query_results = [{"review": "Great product"}]

    state.is_capped = False

    state.generated_sql = sql_generator_factory(unsupported_explanation=None)

    mock_run_prompt.return_value = SimpleNamespace(content="Customers are satisfied.")

    result = node.summarize_review_sentiment(
        state,
        Mock(),
    )

    assert result["answer"] == "Customers are satisfied."

    assert result["chat_history"][0].content == "Customers are satisfied."

    mock_run_prompt.assert_called_once()


def test_summarize_review_sentiment_unsupported(
    node,
    mock_run_prompt,
    sql_generator_factory,
):
    state = Mock()

    state.sanitized_question = "test"
    state.chat_history = []

    state.sanitized_query_results = []
    state.is_capped = False

    state.generated_sql = sql_generator_factory(
        unsupported_explanation="Unsupported question."
    )

    result = node.summarize_review_sentiment(
        state,
        Mock(),
    )

    assert result["answer"] == "Unsupported question."

    assert result["chat_history"][0].content == "Unsupported question."

    mock_run_prompt.assert_not_called()


def test_summarize_review_sentiment_capped(
    node,
    mock_run_prompt,
    sql_generator_factory,
):
    state = Mock()

    state.sanitized_question = "test"
    state.chat_history = []

    state.sanitized_query_results = [{"review": "Great"}]

    state.is_capped = True

    state.generated_sql = sql_generator_factory(
        sql_query="SELECT * FROM reviews",
        unsupported_explanation=None,
    )

    mock_run_prompt.return_value = SimpleNamespace(content="Mostly positive reviews.")

    result = node.summarize_review_sentiment(
        state,
        Mock(),
    )

    assert "safety limit" in result["answer"]

    assert "SELECT * FROM reviews" in result["answer"]


def test_summarize_review_sentiment_no_results(
    node,
    mock_run_prompt,
    sql_generator_factory,
):
    state = Mock()

    state.sanitized_question = "test"
    state.chat_history = []

    state.sanitized_query_results = None

    state.is_capped = False

    state.generated_sql = sql_generator_factory(unsupported_explanation=None)

    mock_run_prompt.return_value = SimpleNamespace(content="No reviews found.")

    result = node.summarize_review_sentiment(
        state,
        Mock(),
    )

    assert result["answer"] == "No reviews found."


def test_summarize_review_sentiment_large_results(
    node,
    mock_run_prompt,
    sql_generator_factory,
):
    state = Mock()

    state.sanitized_question = "test"
    state.chat_history = []

    state.sanitized_query_results = [{"review": "x" * 1000}] * 200

    state.is_capped = False

    state.generated_sql = sql_generator_factory(unsupported_explanation=None)

    mock_run_prompt.return_value = SimpleNamespace(content="Summary generated.")

    result = node.summarize_review_sentiment(
        state,
        Mock(),
    )

    assert result["answer"] == "Summary generated."


def test_summarize_review_sentiment_exception(
    node,
    mock_run_prompt,
    sql_generator_factory,
):
    state = Mock()

    state.sanitized_question = "test"
    state.chat_history = []

    state.sanitized_query_results = [{"review": "Great"}]

    state.is_capped = False

    state.generated_sql = sql_generator_factory(unsupported_explanation=None)

    mock_run_prompt.side_effect = Exception("LLM failed")

    with pytest.raises(Exception) as exc:
        node.summarize_review_sentiment(
            state,
            Mock(),
        )

    assert "Failed to generate review summary" in str(exc.value)


def test_persist_history_with_answer(
    node,
    sql_generator_factory,
):
    state = Mock()

    state.user_id = "user-1"
    state.user_email = "test@example.com"

    state.question = "What are customers?"
    state.sanitized_question = "What are customers?"

    state.answer = "These are customers."

    state.tabular_answer = None

    state.generated_sql = sql_generator_factory(unsupported_explanation=None)

    config = {
        "configurable": {
            "thread_id": "thread-1",
        }
    }

    node.db.upsert_chat_thread = Mock()
    node.db.save_message = Mock()

    result = node.persist_history(state, config)

    assert result == {}

    assert node.db.user_id == "user-1"
    assert node.db.user_email == "test@example.com"

    node.db.upsert_chat_thread.assert_called_once_with(
        "thread-1",
        title="What are customers?...",
    )

    assert node.db.save_message.call_count == 2

    node.db.save_message.assert_any_call(
        "thread-1",
        "user",
        "What are customers?",
        "text",
    )

    node.db.save_message.assert_any_call(
        "thread-1",
        "assistant",
        "These are customers.",
        "text",
    )


def test_persist_history_tabular_answer(
    node,
):
    state = Mock()

    state.user_id = "user-1"
    state.user_email = "test@example.com"

    state.question = "Show customers"
    state.sanitized_question = "Show customers"

    state.answer = None

    state.generated_sql = None

    state.tabular_answer = Mock()
    state.tabular_answer.data = [{"id": 1}]
    state.tabular_answer.answer = "Here are results."

    config = {
        "configurable": {
            "thread_id": "thread-1",
        }
    }

    node.db.upsert_chat_thread = Mock()
    node.db.save_message = Mock()

    node.persist_history(state, config)

    node.db.save_message.assert_any_call(
        "thread-1",
        "assistant",
        {
            "data": [{"id": 1}],
            "answer": "Here are results.",
        },
        "dataframe",
    )


def test_persist_history_warning_answer(
    node,
    sql_generator_factory,
):
    state = Mock()

    state.user_id = "user-1"
    state.user_email = "test@example.com"

    state.question = "dangerous"
    state.sanitized_question = "dangerous"

    state.answer = "This request was flagged for safety reasons"

    state.tabular_answer = None

    state.generated_sql = sql_generator_factory(unsupported_explanation=None)

    config = {
        "configurable": {
            "thread_id": "thread-1",
        }
    }

    node.db.upsert_chat_thread = Mock()
    node.db.save_message = Mock()

    node.persist_history(state, config)

    node.db.save_message.assert_any_call(
        "thread-1",
        "assistant",
        "This request was flagged for safety reasons",
        "warning",
    )


def test_persist_history_unsupported_explanation(
    node,
    sql_generator_factory,
):
    state = Mock()

    state.user_id = "user-1"
    state.user_email = "test@example.com"

    state.question = "unsupported"
    state.sanitized_question = "unsupported"

    state.answer = None
    state.tabular_answer = None

    state.generated_sql = sql_generator_factory(
        unsupported_explanation="Cannot answer this."
    )

    config = {
        "configurable": {
            "thread_id": "thread-1",
        }
    }

    node.db.upsert_chat_thread = Mock()
    node.db.save_message = Mock()

    node.persist_history(state, config)

    node.db.save_message.assert_any_call(
        "thread-1",
        "assistant",
        "Cannot answer this.",
        "warning",
    )


def test_persist_history_missing_thread_id(node):
    state = Mock()

    state.user_id = "user-1"

    config = {"configurable": {}}

    node.db.upsert_chat_thread = Mock()
    node.db.save_message = Mock()

    result = node.persist_history(state, config)

    assert result == {}

    node.db.upsert_chat_thread.assert_not_called()
    node.db.save_message.assert_not_called()


def test_persist_history_missing_user_id(node):
    state = Mock()

    state.user_id = None

    config = {
        "configurable": {
            "thread_id": "thread-1",
        }
    }

    node.db.upsert_chat_thread = Mock()
    node.db.save_message = Mock()

    result = node.persist_history(state, config)

    assert result == {}

    node.db.upsert_chat_thread.assert_not_called()
    node.db.save_message.assert_not_called()


def test_persist_history_no_content(
    node,
    sql_generator_factory,
):
    state = Mock()

    state.user_id = "user-1"
    state.user_email = "test@example.com"

    state.question = "test"
    state.sanitized_question = "test"

    state.answer = None
    state.tabular_answer = None

    state.generated_sql = sql_generator_factory(unsupported_explanation=None)

    config = {
        "configurable": {
            "thread_id": "thread-1",
        }
    }

    node.db.upsert_chat_thread = Mock()
    node.db.save_message = Mock()

    node.persist_history(state, config)

    node.db.upsert_chat_thread.assert_called_once()

    node.db.save_message.assert_not_called()


def test_persist_history_exception(
    node,
):
    state = Mock()

    state.user_id = "user-1"
    state.user_email = "test@example.com"

    state.question = "test"
    state.sanitized_question = "test"

    config = {
        "configurable": {
            "thread_id": "thread-1",
        }
    }

    node.db.upsert_chat_thread = Mock(side_effect=Exception("DB failed"))

    result = node.persist_history(state, config)

    assert result == {}
