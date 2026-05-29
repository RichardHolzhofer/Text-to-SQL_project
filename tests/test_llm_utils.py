from unittest.mock import Mock, patch

import pytest

from src.exceptions.exception import (
    LLMInvocationError,
    LLMTemplateError,
)
from src.utils.llm_utils import generate_conversation_title, run_prompt


def test_run_prompt_success(
    config,
    mock_template,
    mock_llm,
    mock_chain,
):
    config.get_smart_llm = Mock(return_value=mock_llm)

    mock_template.__or__ = Mock(return_value=mock_chain)

    with patch(
        "src.utils.llm_utils.get_prompt_template",
        return_value=mock_template,
    ):
        result = run_prompt(
            prompt_name="test_prompt",
            variables={"question": "test"},
            config=config,
        )

    assert result == "response"

    config.get_smart_llm.assert_called_once()

    mock_chain.invoke.assert_called_once_with(
        {"question": "test"},
        config={"metadata": {}},
    )


def test_run_prompt_fast_llm(
    config,
    mock_template,
    mock_llm,
    mock_chain,
):
    config.get_fast_llm = Mock(return_value=mock_llm)

    mock_template.__or__ = Mock(return_value=mock_chain)

    with patch(
        "src.utils.llm_utils.get_prompt_template",
        return_value=mock_template,
    ):
        run_prompt(
            prompt_name="test_prompt",
            variables={},
            config=config,
            use_fast_llm=True,
        )

    config.get_fast_llm.assert_called_once()


def test_run_prompt_model_override(
    config,
    mock_template,
    mock_llm,
    mock_chain,
):
    mock_template.metadata = {
        "config": {
            "model": "gpt-4",
        }
    }

    config.get_llm = Mock(return_value=mock_llm)

    mock_template.__or__ = Mock(return_value=mock_chain)

    with patch(
        "src.utils.llm_utils.get_prompt_template",
        return_value=mock_template,
    ):
        run_prompt(
            prompt_name="test_prompt",
            variables={},
            config=config,
        )

    config.get_llm.assert_called_once_with("gpt-4")


def test_run_prompt_temperature_binding(
    config,
    mock_template,
    mock_llm,
    mock_chain,
):
    mock_template.metadata = {
        "config": {
            "temperature": "0.7",
        }
    }

    config.get_smart_llm = Mock(return_value=mock_llm)

    mock_template.__or__ = Mock(return_value=mock_chain)

    with patch(
        "src.utils.llm_utils.get_prompt_template",
        return_value=mock_template,
    ):
        run_prompt(
            prompt_name="test_prompt",
            variables={},
            config=config,
        )

    mock_llm.bind.assert_called_once_with(temperature=0.7)


def test_run_prompt_structured_output(
    config,
    mock_template,
    mock_llm,
    mock_chain,
):
    config.get_smart_llm = Mock(return_value=mock_llm)

    mock_template.__or__ = Mock(return_value=mock_chain)

    schema = Mock()

    with patch(
        "src.utils.llm_utils.get_prompt_template",
        return_value=mock_template,
    ):
        run_prompt(
            prompt_name="test_prompt",
            variables={},
            config=config,
            output_schema=schema,
        )

    mock_llm.with_structured_output.assert_called_once_with(schema)


def test_run_prompt_chat_history(
    config,
    mock_template,
    mock_llm,
    mock_chain,
):
    config.get_smart_llm = Mock(return_value=mock_llm)

    mock_template.__or__ = Mock(return_value=mock_chain)

    history = [
        Mock(),
        Mock(),
    ]

    with patch(
        "src.utils.llm_utils.get_prompt_template",
        return_value=mock_template,
    ):
        run_prompt(
            prompt_name="test_prompt",
            variables={},
            config=config,
            chat_history=history,
        )

    assert mock_template.messages == history


def test_run_prompt_runnable_config_metadata(
    config,
    mock_template,
    mock_llm,
    mock_chain,
):
    mock_template.metadata = {
        "langfuse_prompt": "prompt-id",
    }

    config.get_smart_llm = Mock(return_value=mock_llm)

    mock_template.__or__ = Mock(return_value=mock_chain)

    with patch(
        "src.utils.llm_utils.get_prompt_template",
        return_value=mock_template,
    ):
        run_prompt(
            prompt_name="test_prompt",
            variables={"q": 1},
            config=config,
            runnable_config={
                "metadata": {
                    "existing": "value",
                }
            },
        )

    mock_chain.invoke.assert_called_once_with(
        {"q": 1},
        config={
            "metadata": {
                "existing": "value",
                "langfuse_prompt": "prompt-id",
            }
        },
    )


def test_run_prompt_template_error(
    config,
):
    with patch(
        "src.utils.llm_utils.get_prompt_template",
        side_effect=Exception("template failed"),
    ):
        with pytest.raises(LLMTemplateError):
            run_prompt(
                prompt_name="test_prompt",
                variables={},
                config=config,
            )


def test_run_prompt_invocation_error(
    config,
    mock_template,
    mock_llm,
    mock_chain,
):
    config.get_smart_llm = Mock(return_value=mock_llm)

    mock_chain.invoke.side_effect = Exception("invoke failed")

    mock_template.__or__ = Mock(return_value=mock_chain)

    with patch(
        "src.utils.llm_utils.get_prompt_template",
        return_value=mock_template,
    ):
        with pytest.raises(LLMInvocationError):
            run_prompt(
                prompt_name="test_prompt",
                variables={},
                config=config,
            )


def test_generate_conversation_title_success(config):
    mock_response = Mock()
    mock_response.content = '  "Sales Analysis"  '

    with patch(
        "src.utils.llm_utils.run_prompt",
        return_value=mock_response,
    ) as mock_run_prompt:
        result = generate_conversation_title(
            config,
            "Show me sales trends",
        )

    assert result == "Sales Analysis"

    mock_run_prompt.assert_called_once_with(
        prompt_name="generate_title",
        variables={
            "question": "Show me sales trends",
        },
        config=config,
        use_fast_llm=True,
    )


def test_generate_conversation_title_raw_response(config):
    with patch(
        "src.utils.llm_utils.run_prompt",
        return_value="Revenue Summary",
    ):
        result = generate_conversation_title(
            config,
            "Revenue question",
        )

    assert result == "Revenue Summary"


def test_generate_conversation_title_fallback(config):
    question = "What are the total sales by category?"

    with patch(
        "src.utils.llm_utils.run_prompt",
        side_effect=Exception("LLM failed"),
    ):
        result = generate_conversation_title(
            config,
            question,
        )

    assert result == f"{question[:25]}..."


def test_generate_conversation_title_fallback_short_question(config):
    question = "Hello"

    with patch(
        "src.utils.llm_utils.run_prompt",
        side_effect=Exception("LLM failed"),
    ):
        result = generate_conversation_title(
            config,
            question,
        )

    assert result == "Hello..."


def test_generate_conversation_title_without_quotes(config):
    mock_response = Mock()
    mock_response.content = "Customer Retention"

    with patch(
        "src.utils.llm_utils.run_prompt",
        return_value=mock_response,
    ):
        result = generate_conversation_title(
            config,
            "Retention question",
        )

    assert result == "Customer Retention"
