from decimal import Decimal
from typing import Any, Dict, List


def process_snowflake_results(
    cursor_description: Any, rows: List[Any]
) -> List[Dict[str, Any]]:
    """
    Converts raw Snowflake cursor rows into a list of dictionaries,
    handling Decimal to float/int conversion for JSON serialization.
    """
    if not cursor_description:
        return []

    columns = [col[0] for col in cursor_description]
    results = []

    for row in rows:
        processed_row = {}
        for k, v in zip(columns, row):
            # Convert Decimals to float/int for JSON serialization
            if isinstance(v, Decimal):
                # If it's a whole number, use int, otherwise float
                processed_row[k] = float(v) if v % 1 != 0 else int(v)
            else:
                processed_row[k] = v
        results.append(processed_row)

    return results


def is_result_empty(results: List[Dict[str, Any]] | None) -> bool:
    """
    Determines if a result set is truly empty or contains only null/zero placeholders.
    Handles the case where Snowflake returns a single row with 0 or None
    (common for COUNT or SUM on empty datasets).
    """
    if not results or len(results) == 0:
        return True

    if len(results) == 1:
        row = results[0]
        values = list(row.values())
        # Treat a single row with only 0, 0.0, or None as "empty"
        if all(v in (0, 0.0, None) for v in values):
            return True

    return False
