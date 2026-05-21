import os

import yaml

from src.exceptions.exception import YAMLProcessingError


def load_and_filter_schema(
    schema_path: str, exclude_tables: list, exclude_columns: list
):
    try:
        if not os.path.exists(schema_path):
            return {}

        with open(schema_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        tables = {}
        for model in data.get("models", []):
            table_name = model.get("name")
            if table_name in exclude_tables:
                continue

            columns = []
            for col in model.get("columns", []):
                col_name = col.get("name")
                if col_name in exclude_columns:
                    continue

                col_type = "TEXT"
                if "date" in col_name or "timestamp" in col_name or "at" in col_name:
                    col_type = "DATE"
                elif "id" in col_name or "pk" in col_name:
                    col_type = "UUID"
                elif any(
                    k in col_name for k in ["value", "amount", "price", "freight"]
                ):
                    col_type = "FLOAT"
                elif any(k in col_name for k in ["score", "days", "length", "total"]):
                    col_type = "INT"

                tags = []
                tests = col.get("tests", [])
                is_pk = False
                is_fk = False
                if tests:
                    for t in tests:
                        if isinstance(t, str) and t == "unique":
                            is_pk = True
                        elif isinstance(t, dict) and "relationships" in t:
                            is_fk = True

                if is_pk:
                    tags.append("PK")
                if is_fk:
                    tags.append("FK")

                columns.append({"name": col_name, "type": col_type, "tags": tags})

            tables[table_name] = {
                "description": model.get("description", "").strip(),
                "columns": columns,
            }
        return tables
    except Exception as e:
        raise YAMLProcessingError(e)


def generate_chat_title(llm, question: str) -> str:
    """Uses the provided LLM to generate a short title for the chat conversation."""
    try:
        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate.from_template(
            "Generate a short title (maximum 5 words) for a chat conversation that starts with the following question: '{question}'\nTitle:"
        )
        chain = prompt | llm
        response = chain.invoke({"question": question})
        title = response.content.strip().replace('"', "")
        if len(title.split()) > 7:
            title = " ".join(title.split()[:5]) + "..."
        return title
    except Exception:
        return question[:30] + "..."
