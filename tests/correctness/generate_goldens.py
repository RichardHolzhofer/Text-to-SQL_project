import os

import yaml
from deepeval.dataset import EvaluationDataset, Golden

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIOS_DIR = os.path.join(BASE_DIR, "data", "scenarios")
DATASET_FILE = os.path.join(BASE_DIR, "data", "correctness_dataset.json")

# Default scenario definitions (for bootstrapping)
DEFAULT_SCENARIOS = [
    {
        "id": "NL-01",
        "category": "Standard NL",
        "input": "Summarize the customer distribution across different Brazilian states.",
        "limit_type": "standard_nl_limit",
        "intent": "nl",
    },
    {
        "id": "NL-02",
        "category": "Standard NL",
        "input": "Explain the difference in average order value between new and returning customers.",
        "limit_type": "standard_nl_limit",
        "intent": "nl",
    },
    {
        "id": "NL-03",
        "category": "Standard NL",
        "input": "Provide a summary of revenue trends for the first quarter of 2018.",
        "limit_type": "standard_nl_limit",
        "intent": "nl",
    },
    {
        "id": "NL-04",
        "category": "Standard NL",
        "input": "Compare the delivery performance and delay metrics between 'SP' and 'RJ' states.",
        "limit_type": "standard_nl_limit",
        "intent": "nl",
    },
    {
        "id": "NL-05",
        "category": "Standard NL",
        "input": "Summarize the top-performing product categories in terms of total sales volume.",
        "limit_type": "standard_nl_limit",
        "intent": "nl",
    },
    {
        "id": "TAB-01",
        "category": "Standard Tab",
        "input": "Give me a table of the top 5 cities with the most customers.",
        "limit_type": "standard_tab_limit",
        "intent": "tab",
    },
    {
        "id": "TAB-02",
        "category": "Standard Tab",
        "input": "Show a table with the count of orders for each current order status including invalid orders.",
        "limit_type": "standard_tab_limit",
        "intent": "tab",
    },
    {
        "id": "TAB-03",
        "category": "Standard Tab",
        "input": "List the top 5 customers by lifetime value, and include their city, state, and the total number of distinct product categories they have shopped from.",
        "limit_type": "standard_tab_limit",
        "intent": "tab",
    },
    {
        "id": "TAB-04",
        "category": "Standard Tab",
        "input": "List the top 3 cities based on total revenue generated from items sold in the 'watches gifts' category to 'returning customers'.",
        "limit_type": "standard_tab_limit",
        "intent": "tab",
    },
    {
        "id": "TAB-05",
        "category": "Standard Tab",
        "input": "Show a comparison table of the average review score for valid orders that used a voucher versus those that didn't, for the top 5 states with the most valid orders.",
        "limit_type": "standard_tab_limit",
        "intent": "tab",
    },
    {
        "id": "SEM-01",
        "category": "Semantic",
        "input": "What are the most common complaints customers have about shipping and delivery?",
        "limit_type": "semantic_nl_limit",
        "intent": "semantic_nl",
    },
    {
        "id": "SEM-02",
        "category": "Semantic",
        "input": "Summarize the positive feedback specifically for 'electronics' products.",
        "limit_type": "semantic_nl_limit",
        "intent": "semantic_nl",
    },
    {
        "id": "SEM-03",
        "category": "Semantic",
        "input": "Give me a table of 5 reviews that specifically mention 'damaged' or 'broken' items.",
        "limit_type": "semantic_tab_limit",
        "intent": "semantic_tab",
    },
    {
        "id": "SEM-04",
        "category": "Semantic",
        "input": "What is the general feedback regarding the quality of 'health beauty' products?",
        "limit_type": "semantic_nl_limit",
        "intent": "semantic_nl",
    },
    {
        "id": "SEM-05",
        "category": "Semantic",
        "input": "Find and summarize reviews that mention issues with 'customer service' or 'refunds'.",
        "limit_type": "semantic_nl_limit",
        "intent": "semantic_nl",
    },
    {
        "id": "BND-01",
        "category": "Boundaries",
        "input": "List every single order ID and its status in the database.",
        "limit_type": "safety_limit",
        "intent": "tab",
    },
    {
        "id": "BND-02",
        "category": "Boundaries",
        "input": "Summarize the individual order details for all customers in Sao Paulo during 2017, including product and pricing information.",
        "limit_type": "standard_nl_limit",
        "intent": "nl",
    },
    {
        "id": "BND-03",
        "category": "Boundaries",
        "input": "Show a table of the first 50 customers sorted by registration date, including their unique ID, city, state, first purchase date, and lifetime value.",
        "limit_type": "standard_tab_limit",
        "intent": "tab",
    },
    {
        "id": "BND-04",
        "category": "Boundaries",
        "input": "Summarize all reviews that mention the word 'good'.",
        "limit_type": "semantic_nl_limit",
        "intent": "semantic_nl",
    },
    {
        "id": "BND-05",
        "category": "Boundaries",
        "input": "Give me a table of all reviews that mention 'delivery'.",
        "limit_type": "semantic_tab_limit",
        "intent": "semantic_tab",
    },
    {
        "id": "FZY-01",
        "category": "Fuzzy Matching",
        "input": "How many orders were placed by customers in Sao Palo?",
        "limit_type": "standard_tab_limit",
        "intent": "tab",
    },
    {
        "id": "FZY-02",
        "category": "Fuzzy Matching",
        "input": "List all items sold in the 'bed bath tabl' category.",
        "limit_type": "standard_tab_limit",
        "intent": "tab",
    },
    {
        "id": "FZY-03",
        "category": "Fuzzy Matching",
        "input": "What is the total revenue from customers in the state of Minias Gerais?",
        "limit_type": "standard_tab_limit",
        "intent": "tab",
    },
    {
        "id": "FZY-04",
        "category": "Fuzzy Matching",
        "input": "Show the top 3 cities with the most orders for 'beuty' products.",
        "limit_type": "standard_tab_limit",
        "intent": "tab",
    },
    {
        "id": "FZY-05",
        "category": "Fuzzy Matching",
        "input": "Summarize the average delivery time for customers in 'Rio de Janero' for orders with a 'shippd' status.",
        "limit_type": "standard_tab_limit",
        "intent": "tab",
    },
]


def bootstrap_scenarios():
    """Creates the initial YAML files if the directory is empty."""
    os.makedirs(SCENARIOS_DIR, exist_ok=True)

    for item in DEFAULT_SCENARIOS:
        file_path = os.path.join(SCENARIOS_DIR, f"{item['id']}.yaml")
        if not os.path.exists(file_path):
            data = {
                "id": item["id"],
                "category": item["category"],
                "input": item["input"],
                "ground_truth_sql": "",
                "expected_output": f"[Paste your perfect output for {item['id']} here]",
                "metadata": {
                    "limit_type": item["limit_type"],
                    "intent": item["intent"],
                    "download_button": False,
                    "is_capped": False,
                    "is_editdistance": False,
                },
            }
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    data,
                    f,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                )
    print(f"Bootstrapped {len(DEFAULT_SCENARIOS)} YAML files in {SCENARIOS_DIR}")


def generate_dataset():
    """
    Combines YAML scenarios into a DeepEval EvaluationDataset.
    """
    bootstrap_scenarios()

    goldens = []
    files = [f for f in os.listdir(SCENARIOS_DIR) if f.endswith(".yaml")]

    print(f"Scanning {len(files)} YAML files...")

    for filename in files:
        with open(os.path.join(SCENARIOS_DIR, filename), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

            expected = data.get("expected_output", "")
            if not expected or expected.startswith("[Paste your"):
                print(f"Skipping {data['id']}: No manual expected_output found.")
                continue

            # Create Golden with audit metadata
            golden = Golden(
                input=data["input"],
                expected_output=expected,
                name=data["id"],
                additional_metadata={
                    "category": data["category"],
                    "ground_truth_sql": data.get("ground_truth_sql", ""),
                    **data.get("metadata", {}),
                },
            )
            goldens.append(golden)

    if not goldens:
        print(
            "\nNo goldens generated yet. Please populate the 'expected_output' in your YAML files."
        )
        return

    dataset = EvaluationDataset(goldens=goldens)
    dataset.save_as(
        file_type="json",
        directory=os.path.join(BASE_DIR, "data"),
        file_name="correctness_dataset",
    )
    print(f"\nSuccessfully generated dataset with {len(goldens)} cases!")
    print(f"Dataset path: {DATASET_FILE}")


if __name__ == "__main__":
    generate_dataset()
