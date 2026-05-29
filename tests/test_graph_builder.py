from unittest.mock import Mock


def test_route_after_intent_semantic(graph):
    state = Mock()
    state.router = Mock(is_review_query=True, is_semantic_intent=True)

    assert graph._route_after_intent(state) == "semantic"


def test_route_after_intent_general(graph):
    state = Mock()
    state.router = Mock(is_review_query=False, is_semantic_intent=True)

    assert graph._route_after_intent(state) == "general"


def test_route_after_input_guardrail_blocked(graph):
    state = Mock()
    state.answer = "flagged for safety reasons: prompt injection"

    assert graph._route_after_input_guardrail(state) == "blocked"


def test_route_after_input_guardrail_continue(graph):
    state = Mock()
    state.answer = "normal answer"

    assert graph._route_after_input_guardrail(state) == "continue"


def test_route_after_conversation_general(graph):
    state = Mock(is_general_conversation=True)
    assert graph._route_after_conversation_evaluator(state) == "general_conversation"


def test_route_after_conversation_sql(graph):
    state = Mock(is_general_conversation=False)
    assert graph._route_after_conversation_evaluator(state) == "sql_routing"


def test_route_after_validator_retry_standard(graph):
    state = Mock()
    state.validator = Mock(is_valid_query=False, iteration_count=1)
    state.router = Mock(is_review_query=False)

    assert graph._route_after_validator(state) == "standard_generator"


def test_route_after_validator_retry_semantic(graph):
    state = Mock()
    state.validator = Mock(is_valid_query=False, iteration_count=2)
    state.router = Mock(is_review_query=True)

    assert graph._route_after_validator(state) == "semantic_generator"


def test_route_after_validator_execute(graph):
    state = Mock()
    state.validator = Mock(is_valid_query=True)

    assert graph._route_after_validator(state) == "execute"


def test_route_after_execute_anonymize(graph):
    state = Mock()
    state.validator = Mock(is_valid_query=True)

    assert graph._route_after_execute(state) == "anonymize"


def test_route_to_answer_tabular(graph):
    state = Mock()
    state.router = Mock(route="tab", is_review_query=False, is_semantic_intent=False)

    assert graph._route_to_answer(state) == "tabular"


def test_route_to_answer_nl(graph):
    state = Mock()
    state.router = Mock(route="nl", is_review_query=False, is_semantic_intent=False)

    assert graph._route_to_answer(state) == "nl"


def test_route_to_answer_review(graph):
    state = Mock()
    state.router = Mock(route="nl", is_review_query=True, is_semantic_intent=True)

    assert graph._route_to_answer(state) == "review_nl"


def test_build_graph_returns_compiled_graph(graph):
    fake_graph = Mock()
    graph.nodes.build_graph = Mock(return_value=fake_graph)

    result = graph.build_graph()

    assert result is not None
