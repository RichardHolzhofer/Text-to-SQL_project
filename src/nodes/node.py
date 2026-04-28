import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.states.state import Table, RelationshipDigest, Schema, SQLGenerator
from src.utils.utils import load_prompt, load_yaml, dump_yaml
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
            self.llm = init_chat_model("groq:openai/gpt-oss-120b")
            self.max_retry = 3
            logger.info("TextToSQLNodes initialized with LLM and Config.")
        except Exception as e:
            logger.exception("Failed to initialize TextToSQLNodes.")
            raise NodeException(e)

    def _convert_to_messages(self, base_messages: list):
        """Helper to convert tuple-based prompts into LangChain Message objects."""
        langchain_messages = []
        for role, content in base_messages:
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
        return langchain_messages

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
            logger.info(
                f"Building unified schema from {mart_schema_path} and {enhancement_schema_path}"
            )

            self.table_extractor = self.llm.with_structured_output(Table)
            self.relationship_extractor = self.llm.with_structured_output(
                RelationshipDigest
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

                # Load the prompt for table extraction
                extract_table_prompt = load_prompt(
                    "src/prompts/extract_table.yaml",
                    {"model_yaml": model_yaml_str},
                )

                extract_table_prompt_messages = self._convert_to_messages(
                    extract_table_prompt
                )
                table_schema = self.table_extractor.invoke(
                    extract_table_prompt_messages
                )
                db_tables.append(table_schema)

            logger.info(f"Successfully extracted {len(db_tables)} table definitions.")

            # Extract relationships using the unified context
            logger.info("Starting relationship extraction across all tables...")
            relationship_context = {
                "source_yaml": mart_schema_yaml,
                "tables": [table.model_dump() for table in db_tables],
            }

            extract_relationships_prompt = load_prompt(
                "src/prompts/extract_relationships.yaml",
                {"relationship_context": json.dumps(relationship_context, indent=2)},
            )

            extract_relationships_prompt_messages = self._convert_to_messages(
                extract_relationships_prompt
            )
            relationship_digest = self.relationship_extractor.invoke(
                extract_relationships_prompt_messages
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
            return {"schema": final_schema}

        except Exception as e:
            logger.exception("Failed to build unified schema.")
            raise SchemaBuildError(e)

    def generate_sql(self, state: TextToSQLState):
        try:
            logger.info(
                f"Generating SQL query for question: '{state.question}' (Iteration: {state.iteration_count})"
            )

            # Bind the LLM to our SQLGenerator schema
            sql_extractor = self.llm.with_structured_output(SQLGenerator)

            # Serialize the schema so the LLM can read it
            schema_json = json.dumps(state.schema.model_dump(), indent=2)

            # Load the prompt with dynamic context
            generate_sql_prompt = load_prompt(
                "src/prompts/generate_sql.yaml",
                {"schema_context": schema_json, "question": state.question},
            )

            # Convert loaded prompt into LangChain message objects
            generate_sql_prompt_messages = self._convert_to_messages(
                generate_sql_prompt
            )

            # Append real conversational history (if any)
            if state.chat_history:
                generate_sql_prompt_messages.extend(state.chat_history)

            # Execute
            generated = sql_extractor.invoke(generate_sql_prompt_messages)

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
                    "error_message": None,  # No Snowflake error, just unsupported
                }

            logger.info(
                f"Validating SQL via EXPLAIN (Iteration: {state.iteration_count + 1})"
            )

            # Execute EXPLAIN in Snowflake
            with self.config.get_connection(write_access=False) as conn:
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
