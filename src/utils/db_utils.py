import time
from decimal import Decimal
from typing import Any, Callable, Dict, List

from src.exceptions.exception import (
    NetworkRetryExhaustedError,
    SnowflakeResultProcessingError,
)


def retry_transient_network(
    operation: Callable[[], Any],
    attempts: int = 4,
    base_delay_sec: float = 0.35,
) -> Any:
    """
    Re-run an operation when Supabase/httpx hits flaky TLS or dropped connections
    (e.g. SSL: UNEXPECTED_EOF_WHILE_READING from a container).
    """
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as e:
            last_error = e
            if attempt >= attempts:
                break
            time.sleep(base_delay_sec * (2 ** (attempt - 1)))

    if last_error is None:
        raise NetworkRetryExhaustedError(
            RuntimeError("retry_transient_network: no attempts were run")
        )
    if isinstance(last_error, Exception):
        raise NetworkRetryExhaustedError(last_error) from last_error
    raise last_error


def process_snowflake_results(
    cursor_description: Any, rows: List[Any]
) -> List[Dict[str, Any]]:
    """
    Converts raw Snowflake cursor rows into a list of dictionaries,
    handling Decimal to float/int conversion for JSON serialization.
    """
    try:
        if not cursor_description:
            return []

        columns = [col[0] for col in cursor_description]
        results = []

        for row in rows:
            processed_row = {}
            for k, v in zip(columns, row):
                if isinstance(v, Decimal):
                    processed_row[k] = float(v) if v % 1 != 0 else int(v)
                else:
                    processed_row[k] = v
            results.append(processed_row)

        return results
    except Exception as e:
        raise SnowflakeResultProcessingError(e) from e


def is_result_empty(results: List[Dict[str, Any]] | None) -> bool:
    """
    Determines if a result set is truly empty or contains only null/zero placeholders.
    Handles the case where Snowflake returns a single row with 0 or None
    (common for COUNT or SUM on empty datasets).
    """
    try:
        if not results or len(results) == 0:
            return True

        if len(results) == 1:
            row = results[0]
            values = list(row.values())
            if all(v in (0, 0.0, None) for v in values):
                return True

        return False
    except (TypeError, AttributeError) as e:
        raise SnowflakeResultProcessingError(e) from e
