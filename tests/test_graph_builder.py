from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.graph_builder.graph_builder import TextToSQLGraph


def make_graph(max_retry=3):
    with patch("src.graph_builder.graph_builder.Config") as mock_config_cls:
        with patch("src.graph_builder.graph_builder.TextToSQLNodes") as mock_nodes_cls:
            mock_config = Mock()
            mock_config_cls.return_value = mock_config
            mock_nodes = Mock()
            mock_nodes.max_retry = max_retry
            mock_nodes_cls.return_value = mock_nodes
            return TextToSQLGraph()


def test_route_after_intent_semantic(sample_state_factory):
    graph = make_graph()
    state = sample_state_factory(
        router=SimpleNamespace(is_review_query=True, is_semantic_intent=True)
    )
    assert graph._route_after_intent(state) == "semantic"


def test_route_after_intent_general(sample_state_factory):
    graph = make_graph()
    state = sample_state_factory(
        router=SimpleNamespace(is_review_query=False, is_semantic_intent=True)
    )
    assert graph._route_after_intent(state) == "general"


def test_route_after_input_guardrail_blocked(sample_state_factory):
    graph = make_graph()
    state = sample_state_factory(answer="This request was flagged for safety reasons.")
    assert graph._route_after_input_guardrail(state) == "blocked"


def test_route_after_input_guardrail_continue(sample_state_factory):
    graph = make_graph()
    state = sample_state_factory(answer="Safe response")
    assert graph._route_after_input_guardrail(state) == "continue"


def test_route_after_conversation_evaluator_general(sample_state_factory):
    graph = make_graph()
    state = sample_state_factory(is_general_conversation=True)
    assert graph._route_after_conversation_evaluator(state) == "general_conversation"


def test_route_after_conversation_evaluator_sql(sample_state_factory):
    graph = make_graph()
    state = sample_state_factory(is_general_conversation=False)
    assert graph._route_after_conversation_evaluator(state) == "sql_routing"


def test_route_after_validator_standard_generator(sample_state_factory):
    graph = make_graph(max_retry=3)
    state = sample_state_factory(
        validator=SimpleNamespace(is_valid_query=False, iteration_count=1),
        router=SimpleNamespace(is_review_query=False),
    )
    assert graph._route_after_validator(state) == "standard_generator"


def test_route_after_validator_semantic_generator(sample_state_factory):
    graph = make_graph(max_retry=3)
    state = sample_state_factory(
        validator=SimpleNamespace(is_valid_query=False, iteration_count=2),
        router=SimpleNamespace(is_review_query=True),
    )
    assert graph._route_after_validator(state) == "semantic_generator"


def test_route_after_validator_execute(sample_state_factory):
    graph = make_graph(max_retry=3)
    state = sample_state_factory(
        validator=SimpleNamespace(is_valid_query=True, iteration_count=0),
        router=SimpleNamespace(is_review_query=False),
    )
    assert graph._route_after_validator(state) == "execute"


def test_route_after_execute_standard_generator(sample_state_factory):
    graph = make_graph(max_retry=3)
    state = sample_state_factory(
        validator=SimpleNamespace(is_valid_query=False, iteration_count=1),
        router=SimpleNamespace(is_review_query=False),
    )
    assert graph._route_after_execute(state) == "standard_generator"


def test_route_after_execute_anonymize(sample_state_factory):
    graph = make_graph(max_retry=3)
    state = sample_state_factory(
        validator=SimpleNamespace(is_valid_query=True, iteration_count=0),
        router=SimpleNamespace(is_review_query=False),
    )
    assert graph._route_after_execute(state) == "anonymize"


def test_route_to_answer_review_nl(sample_state_factory):
    graph = make_graph()
    state = sample_state_factory(
        router=SimpleNamespace(
            is_review_query=True,
            route="nl",
            is_semantic_intent=True,
        )
    )
    assert graph._route_to_answer(state) == "review_nl"


def test_route_to_answer_tabular(sample_state_factory):
    graph = make_graph()
    state = sample_state_factory(
        router=SimpleNamespace(
            is_review_query=False,
            route="tab",
            is_semantic_intent=False,
        )
    )
    assert graph._route_to_answer(state) == "tabular"


def test_route_to_answer_nl_default(sample_state_factory):
    graph = make_graph()
    state = sample_state_factory(
        router=SimpleNamespace(
            is_review_query=False,
            route="nl",
            is_semantic_intent=False,
        )
    )
    assert graph._route_to_answer(state) == "nl"
