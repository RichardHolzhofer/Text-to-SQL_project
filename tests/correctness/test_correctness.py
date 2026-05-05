import os
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

    # Capture the actual output (NL answer or Tabular answer description)
    actual_output = result.get("answer") or "No answer generated."

    # If it was a tabular request, the NL answer usually contains the description/disclaimer
    # and the tabular_answer object contains the actual data.
    if result.get("tabular_answer"):
        actual_output = result["tabular_answer"].answer

    # Create the test case for DeepEval
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
        additional_metadata=golden.additional_metadata,  # Includes ID, Category, SQL etc.
    )

    # Evaluate
    assert_test(test_case, [correctness_metric])
