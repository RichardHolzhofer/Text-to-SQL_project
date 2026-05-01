import json
from langchain_core.messages import AIMessage

from src.states.state import Table, RelationshipDigest, Schema, Router, SQLGenerator
from src.utils.utils import dump_yaml, load_yaml
from src.utils.llm_utils import run_prompt
from src.logger.logger import logger
from src.config.config import Config
from src.exceptions.exception import SchemaBuildError, NodeException, SQLGenerationError
from src.states.state import TextToSQLState


class TextToSQLNodes:
    def __init__(self, config: Config):
        """
        Initialize the node handler with shared configuration.

        Args:
            config (Config): The centralized configuration object.
        """
        try:
            self.config = config
            self.smart_llm = config.get_smart_llm()
            self.fast_llm = config.get_fast_llm()
            self.max_retry = 3
            logger.info("TextToSQLNodes initialized using Config LLMs.")
        except Exception as e:
            logger.exception("Failed to initialize TextToSQLNodes.")
            raise NodeException(e)

    def build_schema(
        self,
        state: TextToSQLState,
        mart_schema_path="ecommerce_analytics/models/dbt_mrt/_dbt_mrt_schema.yml",
        enhancement_schema_path="embeddings/_embeddings_schema.yml",
    ):
        """
        Gathers dbt metadata and embeddings metadata to create a unified
        Schema for the LLM.
        """
        try:
            # Reset intermediate state variables for each new turn
            reset_state = {
                "intent": None,
                "generated_sql": None,
                "is_valid_query": None,
                "error_message": None,
                "iteration_count": 0,
                "query_results": None,
                "answer": None,
                "tabular_answer": None,
            }

            if state.schema is not None:
                logger.info("Schema is already loaded in state. Skipping extraction.")
                reset_state["schema"] = state.schema
                return reset_state

            logger.info(
                f"Building unified schema from {mart_schema_path} and {enhancement_schema_path}"
            )

            # Load raw YAML files
            mart_schema_yaml = load_yaml(mart_schema_path)
            enhancement_schema_yaml = load_yaml(enhancement_schema_path)

            # Merge the model lists to incorporate embeddings
            unified_model_list = mart_schema_yaml.get("models", [])
            unified_model_list.extend(enhancement_schema_yaml.get("models", []))

            logger.info(f"Identified {len(unified_model_list)} models to extract.")

            db_tables = []

            # Extract Table objects from each model
            for model in unified_model_list:
                model_name = model.get("name", "unknown")
                logger.info(f"Extracting structured metadata for table: {model_name}")

                model_yaml_str = dump_yaml({"model": model})

                # Run structured extraction
                table_schema = run_prompt(
                    prompt_name="extract_table",
                    variables={"model_yaml": model_yaml_str},
                    config=self.config,
                    output_schema=Table,
                )
                db_tables.append(table_schema)

            logger.info(f"Successfully extracted {len(db_tables)} table definitions.")

            # Extract relationships using the unified context
            logger.info("Starting relationship extraction across all tables...")
            relationship_context = {
                "tables": [table.model_dump() for table in db_tables],
            }

            # Run structured extraction
            relationship_digest = run_prompt(
                prompt_name="extract_relationships",
                variables={
                    "relationship_context": json.dumps(relationship_context, indent=2)
                },
                config=self.config,
                output_schema=RelationshipDigest,
            )

            logger.info(
                f"Extracted {len(relationship_digest.relationships)} foreign key relationships."
            )

            # Build the final unified Schema state
            final_schema = Schema(
                tables=db_tables,
                relationships=relationship_digest.relationships,
            )

            logger.info("Unified schema build complete.")
            reset_state["schema"] = final_schema
            return reset_state

        except Exception as e:
            logger.exception("Failed to build unified schema.")
            raise SchemaBuildError(e)

    def route_format(self, state: TextToSQLState):
        """
        Uses an LLM to determine if the user wants tabular or natural language format.
        """
        try:
            # Run structured routing
            response = run_prompt(
                prompt_name="evaluate_intent",
                variables={"question": state.question},
                config=self.config,
                output_schema=Router,
                use_fast_llm=True,
            )

            return {"intent": response.route}

        except Exception:
            logger.exception("Failed to route format.")
            # We still need a default fallback in case of API/Network errors
            return {"intent": "nl"}

    def generate_sql(self, state: TextToSQLState):
        try:
            logger.info(
                f"Generating SQL query for question: '{state.question}' (Iteration: {state.iteration_count + 1})"
            )

            # Serialize the schema so the LLM can read it
            schema_json = json.dumps(state.schema.model_dump(), indent=2)

            # Run structured generation
            generated = run_prompt(
                prompt_name="generate_sql",
                variables={"schema_context": schema_json, "question": state.question},
                config=self.config,
                chat_history=state.chat_history,
                output_schema=SQLGenerator,
            )

            logger.info("SQL generation successful.")
            logger.info(
                f"Thought Process snippet: {generated.thought_process[:100]}..."
            )

            return {"generated_sql": generated}

        except Exception as e:
            logger.exception("Failed to create SQL query.")
            raise SQLGenerationError(e)

    def validate_sql(self, state: TextToSQLState):
        """
        Validates the generated SQL using Snowflake's EXPLAIN command.
        This checks for syntax and object existence without executing the query.
        """
        try:
            if not state.generated_sql:
                raise ValueError("No SQL generated to validate.")

            sql = state.generated_sql.sql_query

            # If the model explicitly said it can't answer, don't run EXPLAIN and mark as invalid
            if sql is None:
                logger.info(
                    "No SQL generated (unsupported question). Skipping EXPLAIN validation."
                )
                return {
                    "is_valid_query": False,
                    "error_message": None,
                }

            logger.info(
                f"Validating SQL via EXPLAIN (Iteration: {state.iteration_count + 1})"
            )

            # Execute EXPLAIN in Snowflake
            with self.config.get_snowflake_connection(write_access=False) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"EXPLAIN {sql}")

            logger.info("SQL validation successful.")
            return {
                "is_valid_query": True,
                "error_message": None,
            }

        except Exception as e:
            error_str = str(e)
            logger.warning(f"SQL validation failed: {error_str}")
            return {
                "is_valid_query": False,
                "error_message": error_str,
                "iteration_count": state.iteration_count + 1,
            }

    def execute_sql(self, state: TextToSQLState):
        """
        Executes the validated SQL query against Snowflake.
        Fetches the results and stores them in the state.
        """
        try:
            if (
                not state.is_valid_query
                or not state.generated_sql
                or not state.generated_sql.sql_query
            ):
                logger.info(
                    "Skipping execution: query is either invalid or unsupported."
                )
                return {"query_results": None}

            sql = state.generated_sql.sql_query
            logger.info("Executing SQL query in Snowflake...")

            results = []
            with self.config.get_snowflake_connection(write_access=False) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    if cursor.description:
                        columns = [col[0] for col in cursor.description]
                        rows = cursor.fetchall()
                        for row in rows:
                            results.append(dict(zip(columns, row)))

            logger.info(f"Execution successful. Fetched {len(results)} rows.")
            return {"query_results": results}

        except Exception as e:
            logger.exception("Failed to execute SQL query.")
            raise NodeException(f"Execution failed: {e}")

    def generate_tabular_answer(self, state: TextToSQLState):
        """
        Simply passes the query results to the tabular_answer state field.
        """
        logger.info("Generating tabular answer...")
        return {
            "tabular_answer": state.query_results,
            "chat_history": [
                AIMessage(content="Here are the tabular results you requested.")
            ],
        }

    def generate_nl_answer(self, state: TextToSQLState):
        """
        Uses an LLM to generate a natural language summary of the query results.
        """
        try:
            logger.info("Generating natural language answer...")

            # If the query was unsupported, just return the explanation as the answer
            if state.generated_sql and state.generated_sql.unsupported_explanation:
                explanation = state.generated_sql.unsupported_explanation
                return {
                    "answer": explanation,
                    "chat_history": [AIMessage(content=explanation)],
                }

            # Be mindful of result size. If it's too large, we might need to truncate.
            results_str = json.dumps(state.query_results, indent=2, default=str)
            if len(results_str) > 50000:
                results_str = results_str[:50000] + "\n... [TRUNCATED]"

            # Run NL generation
            response = run_prompt(
                prompt_name="generate_nl_answer",
                variables={
                    "question": state.question,
                    "sql_query_results": results_str,
                },
                config=self.config,
                chat_history=state.chat_history,
                use_fast_llm=True,
            )
            answer = response.content.strip()

            return {"answer": answer, "chat_history": [AIMessage(content=answer)]}

        except Exception as e:
            logger.exception("Failed to generate natural language answer.")
            raise NodeException(f"Failed to generate NL answer: {e}")
