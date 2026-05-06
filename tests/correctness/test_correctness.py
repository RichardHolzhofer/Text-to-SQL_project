import os
import re
import uuid

import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from src.config.config import Config
from src.database.db import SupabaseDB
from src.graph_builder.graph_builder import TextToSQLGraph
from src.states.state import Schema


# 1. SETUP: Initialize Graph and Cache Schema once per session
@pytest.fixture(scope="session")
def graph():
    return TextToSQLGraph().build_graph()


@pytest.fixture(scope="session")
def cached_schema():
    config = Config()
    db = SupabaseDB(config, admin=True)
    schema_data = db.get_schema_cache("unified_schema")
    if not schema_data:
        pytest.fail(
            "Schema cache not found in Supabase. Please run the app once to build it."
        )
    return Schema.model_validate(schema_data)


# 2. METRIC: Define Factual Correctness via G-Eval
correctness_metric = GEval(
    name="Factual Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
)

# 3. DATASET: Load the generated Goldens
dataset = EvaluationDataset.from_json(
    os.path.join(os.path.dirname(__file__), "data", "correctness_dataset.json")
)


def normalize_text(text: str) -> str:
    """Removes markdown formatting and normalizes whitespace for cleaner comparison."""
    if not text:
        return ""
    # Remove markdown bolding and italics
    text = text.replace("**", "").replace("*", "")
    # Remove list markers at the start of lines
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 4. TEST EXECUTION
@pytest.mark.parametrize("golden", dataset.goldens)
def test_text_to_sql_correctness(golden, graph, cached_schema):
    # Run the graph with a fresh thread_id for each case
    config_run = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Pass the pre-cached schema to bypass extraction logic
    initial_state = {
        "question": golden.input,
        "schema": cached_schema,
        "force_refresh": False,
    }

    print(f"\nRunning test for: {golden.input}")
    result = graph.invoke(initial_state, config=config_run)

    # 3. Extract Outputs based on Intent
    metadata = golden.additional_metadata
    intent = metadata.get("intent", "nl")

    if "tab" in intent:
        # For tabular, we evaluate the 'answer' disclaimer text
        actual_output = (
            result.get("tabular_answer").answer if result.get("tabular_answer") else ""
        )
        # Check is_capped logic
        expected_capped = metadata.get("is_capped", False)
        actual_capped = (
            result.get("tabular_answer").is_capped
            if result.get("tabular_answer")
            else False
        )
        assert actual_capped == expected_capped, (
            f"Capping mismatch for {metadata.get('id')}"
        )
    else:
        actual_output = result.get("answer") or "No answer generated."

    # 4. Validate Download Button mention
    expected_download = metadata.get("download_button", False)
    has_download_mention = "download button" in actual_output.lower()
    if expected_download:
        assert has_download_mention, (
            f"Model forgot to mention download button in {metadata.get('id')}"
        )
    else:
        assert not has_download_mention, (
            f"Model hallucinated a download button in {metadata.get('id')}"
        )

    # Create the test case for DeepEval with Normalization
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=normalize_text(actual_output),
        expected_output=normalize_text(golden.expected_output),
        additional_metadata=metadata,
    )

    # Evaluate
    assert_test(test_case, [correctness_metric])
