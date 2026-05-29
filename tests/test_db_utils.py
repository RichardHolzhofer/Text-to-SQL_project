from decimal import Decimal
from unittest.mock import Mock, call, patch

import pytest

from src.exceptions.exception import (
    NetworkRetryExhaustedError,
    SnowflakeResultProcessingError,
)
from src.utils.db_utils import (
    is_result_empty,
    process_snowflake_results,
    retry_transient_network,
)


def test_retry_transient_network_success_first_try():
    operation = Mock(return_value="success")

    result = retry_transient_network(operation)

    assert result == "success"

    operation.assert_called_once()


def test_retry_transient_network_retry_then_success():
    operation = Mock(
        side_effect=[
            Exception("temporary failure"),
            "success",
        ]
    )

    with patch("src.utils.db_utils.time.sleep") as mock_sleep:
        result = retry_transient_network(
            operation,
            attempts=4,
            base_delay_sec=1,
        )

    assert result == "success"

    assert operation.call_count == 2

    mock_sleep.assert_called_once_with(1)


def test_retry_transient_network_exhausted():
    operation = Mock(side_effect=Exception("network failed"))

    with patch("src.utils.db_utils.time.sleep") as mock_sleep:
        with pytest.raises(NetworkRetryExhaustedError):
            retry_transient_network(
                operation,
                attempts=4,
                base_delay_sec=1,
            )

    assert operation.call_count == 4

    assert mock_sleep.call_count == 3


def test_retry_transient_network_exponential_backoff():
    operation = Mock(side_effect=Exception("network failed"))

    with patch("src.utils.db_utils.time.sleep") as mock_sleep:
        with pytest.raises(NetworkRetryExhaustedError):
            retry_transient_network(
                operation,
                attempts=4,
                base_delay_sec=0.5,
            )

    mock_sleep.assert_has_calls(
        [
            call(0.5),
            call(1.0),
            call(2.0),
        ]
    )


def test_retry_transient_network_zero_attempts():
    operation = Mock()

    with pytest.raises(NetworkRetryExhaustedError):
        retry_transient_network(
            operation,
            attempts=0,
        )

    operation.assert_not_called()


def test_process_snowflake_results_success():
    cursor_description = [
        ("id",),
        ("name",),
    ]

    rows = [
        (1, "Alice"),
        (2, "Bob"),
    ]

    result = process_snowflake_results(
        cursor_description,
        rows,
    )

    assert result == [
        {
            "id": 1,
            "name": "Alice",
        },
        {
            "id": 2,
            "name": "Bob",
        },
    ]


def test_process_snowflake_results_decimal_to_int():
    cursor_description = [
        ("amount",),
    ]

    rows = [
        (Decimal("5"),),
    ]

    result = process_snowflake_results(
        cursor_description,
        rows,
    )

    assert result == [
        {
            "amount": 5,
        }
    ]

    assert isinstance(result[0]["amount"], int)


def test_process_snowflake_results_decimal_to_float():
    cursor_description = [
        ("amount",),
    ]

    rows = [
        (Decimal("5.25"),),
    ]

    result = process_snowflake_results(
        cursor_description,
        rows,
    )

    assert result == [
        {
            "amount": 5.25,
        }
    ]

    assert isinstance(result[0]["amount"], float)


def test_process_snowflake_results_empty_description():
    result = process_snowflake_results(
        [],
        [(1, "Alice")],
    )

    assert result == []


def test_process_snowflake_results_mixed_types():
    cursor_description = [
        ("id",),
        ("price",),
        ("active",),
        ("name",),
    ]

    rows = [
        (
            1,
            Decimal("10.50"),
            True,
            "Alice",
        )
    ]

    result = process_snowflake_results(
        cursor_description,
        rows,
    )

    assert result == [
        {
            "id": 1,
            "price": 10.5,
            "active": True,
            "name": "Alice",
        }
    ]


def test_process_snowflake_results_exception():
    cursor_description = Mock(side_effect=Exception("boom"))

    rows = [
        (1, "Alice"),
    ]

    with pytest.raises(SnowflakeResultProcessingError):
        process_snowflake_results(
            cursor_description,
            rows,
        )


def test_is_result_empty_none():
    result = is_result_empty(None)

    assert result is True


def test_is_result_empty_empty_list():
    result = is_result_empty([])

    assert result is True


def test_is_result_empty_single_zero_row():
    results = [
        {
            "count": 0,
        }
    ]

    result = is_result_empty(results)

    assert result is True


def test_is_result_empty_single_float_zero_row():
    results = [
        {
            "sum": 0.0,
        }
    ]

    result = is_result_empty(results)

    assert result is True


def test_is_result_empty_single_none_row():
    results = [
        {
            "value": None,
        }
    ]

    result = is_result_empty(results)

    assert result is True


def test_is_result_empty_placeholder_row():
    results = [
        {
            "count": 0,
            "sum": 0.0,
            "avg": None,
        }
    ]

    result = is_result_empty(results)

    assert result is True


def test_is_result_empty_single_real_row():
    results = [
        {
            "count": 5,
        }
    ]

    result = is_result_empty(results)

    assert result is False


def test_is_result_empty_multiple_rows():
    results = [
        {
            "id": 1,
        },
        {
            "id": 2,
        },
    ]

    result = is_result_empty(results)

    assert result is False


def test_is_result_empty_exception():
    bad_row = Mock()

    bad_row.values.side_effect = AttributeError("values failed")

    results = [bad_row]

    with pytest.raises(SnowflakeResultProcessingError):
        is_result_empty(results)
