import re
import uuid
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from src.config.config import Config
from src.database.db import SupabaseDB
from src.graph_builder.graph_builder import TextToSQLGraph
from src.states.state import Schema
from langfuse.langchain import CallbackHandler
from langfuse import propagate_attributes

# 1. SETUP: Initialize Graph, Langfuse and Cache Schema once per session
langfuse_handler = CallbackHandler()

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
    model='gpt-4.1',
    name="Factual Correctness",
    evaluation_steps=[
        "Check whether the facts in the actual output contradict any facts in the expected output.",
        "Check whether the key data points, numbers, and categories in the actual output are factually accurate based on the expected output.",
        "Do not penalize for variations in column names or keys (e.g., 'AVG_SCORE' vs 'AVERAGE_SCORE') if the meaning and data are equivalent.",
        "Ignore minor formatting differences such as casing or extra spaces if the factual content matches.",
        "Do NOT penalize for omission of detail — the actual output only needs to correctly answer the question, not match the level of detail in the expected output.",
        "Vague language or missing elaboration is acceptable as long as no incorrect facts are stated."
    ],
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
)

# 3. DATASET: Load the generated Goldens
dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(
    str(Path(__file__).parent / "data" / "correctness_dataset.json")
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
    # Run the graph with a fresh thread_id and Langfuse callbacks
    metadata = golden.additional_metadata
    config_run = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "callbacks": [langfuse_handler],
        "metadata": {
            "test_case_id": metadata.get("id"),
            "run_type": "correctness_test"
        },
        "tags": ["test", metadata.get("category", "general")]
    }

    # Pass the pre-cached schema to bypass extraction logic
    initial_state = {
        "question": golden.input,
        "schema": cached_schema,
        "force_refresh": False,
    }

    with propagate_attributes(
        trace_name=f"TEST-{golden.name}",
        session_id=f"eval_session_{uuid.uuid4().hex[:8]}",
        user_id="eval_service",
        metadata={
            "name":golden.name,
            "intent":metadata.get("intent"),
            "limit_type":metadata.get("limit_type")
        }
    ):
        print(f"\nRunning test for: {golden.input}")
        result = graph.invoke(initial_state, config=config_run)

        # 3. Extract Outputs based on Intent
        intent = metadata.get("intent", "nl")

        if "tab" in intent:
            # For tabular, we combine the 'answer' disclaimer with the 'data' list
            tab_res = result.get("tabular_answer")
            if tab_res:
                actual_output = tab_res.answer
                if tab_res.data:
                    import json
                    # Dump data to string to match the expected_output format
                    data_json = json.dumps(tab_res.data, default=str)
                    actual_output = f"{actual_output} {data_json}"
            else:
                actual_output = "No tabular answer generated."
            
            # Check is_capped logic
            expected_capped = metadata.get("is_capped", False)
            actual_capped = tab_res.is_capped if tab_res else False
            assert actual_capped == expected_capped, (
                f"Capping mismatch for {metadata.get('id')}"
            )
        else:
            actual_output = result.get("answer") or "No answer generated."


    # Create the test case for DeepEval with Normalization
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=normalize_text(actual_output),
        expected_output=normalize_text(golden.expected_output),
        additional_metadata=metadata,
    )

    # Evaluate
    assert_test(test_case, [correctness_metric])
