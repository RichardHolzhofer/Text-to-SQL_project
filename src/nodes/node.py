import json
from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage

from src.config.config import Config
from src.database.db import SupabaseDB
from src.exceptions.exception import NodeException, SchemaBuildError, SQLGenerationError
from src.logger.logger import logger
from src.states.state import (
    RelationshipDigest,
    Router,
    Schema,
    SQLGenerator,
    Table,
    TabularResponse,
    TextToSQLState,
    Validator,
)
from src.utils.db_utils import is_result_empty, process_snowflake_results
from src.utils.llm_utils import run_prompt
from src.utils.utils import clean_sql_query, dump_yaml, load_yaml


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
            self.db = SupabaseDB(config, admin=True)
            self.max_retry = 3
            self.semantic_search_threshold = 0.85
            self.semantic_tab_limit = 10
            self.semantic_nl_limit = 100
            self.standard_tab_limit = 10
            self.standard_nl_limit = 100
            self.mart_schema_path = "olist/models/marts/_marts_schema.yml"
            self.enhancement_schema_path = "embeddings/_embeddings_schema.yml"
            self.safety_limit = 5000
            logger.info("TextToSQLNodes initialized using Config LLMs and Supabase.")
        except Exception as e:
            logger.exception("Failed to initialize TextToSQLNodes.")
            raise NodeException(e)

    def build_schema(
        self,
        state: TextToSQLState,
    ):
        """
        Gathers dbt metadata and embeddings metadata to create a unified
        Schema for the LLM.
        """
        try:
            # Reset intermediate state variables for each new turn
            reset_state = {
                "router": None,
                "generated_sql": None,
                "validator": Validator(),
                "query_results": None,
                "answer": None,
                "tabular_answer": None,
            }

            if state.schema is not None:
                logger.info("Schema is already loaded in state. Skipping extraction.")
                reset_state["schema"] = state.schema
                return reset_state

            # 2. Check Supabase Cache (Unless force_refresh is True)
            if not state.force_refresh:
                cache_data = self.db.get_schema_cache("unified_schema")
                if cache_data:
                    logger.info("Schema found in Supabase cache. Loading...")
                    try:
                        final_schema = Schema.model_validate(cache_data)
                        reset_state["schema"] = final_schema
                        return reset_state
                    except Exception as e:
                        logger.warning(
                            f"Failed to validate cached schema: {e}. Rebuilding."
                        )
            else:
                logger.info(
                    "force_refresh is True. Ignoring cache and rebuilding schema."
                )

            logger.info(
                f"Building unified schema from {self.mart_schema_path} and {self.enhancement_schema_path}"
            )

            # Load raw YAML files
            mart_schema_yaml = load_yaml(self.mart_schema_path)
            enhancement_schema_yaml = load_yaml(self.enhancement_schema_path)

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
                updated_at=datetime.now().isoformat(),
            )

            logger.info("Unified schema build complete.")

            # 4. Save to Supabase Cache
            logger.info("Saving newly built schema to Supabase cache...")
            self.db.insert_schema_cache("unified_schema", final_schema.model_dump())

            reset_state["schema"] = final_schema
            return reset_state

        except Exception as e:
            logger.exception("Failed to build unified schema.")
            raise SchemaBuildError(e)

    def route_format(self, state: TextToSQLState):
        """
        Uses an LLM to determine if the user wants tabular or natural language format,
        and also detects review and semantic intents.
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

            return {"router": response}

        except Exception:
            logger.exception("Failed to route format.")
            # Default fallback
            return {"router": Router(route="nl")}

    def semantic_query_generator(self, state: TextToSQLState):
        """
        Generates the specialized vector SQL using the semantic prompt.
        """
        try:
            logger.info(
                f"Generating Semantic SQL query for question: '{state.question}'"
            )

            # Generate search concept for embedding
            logger.info("Extracting search concept from question...")
            concept_response = run_prompt(
                prompt_name="extract_search_concept",
                variables={"question": state.question},
                config=self.config,
                use_fast_llm=True,
            )
            search_concept = concept_response.content.strip()
            logger.info(f"Extracted concept: '{search_concept}'")

            # Generate embedding for the extracted concept
            logger.info(f"Generating embedding for concept: '{search_concept}'")
            embedding_model = self.config.get_embedding_model()
            query_vector = embedding_model.embed_query(search_concept)

            # Serialize the schema so the LLM can read it
            schema_json = json.dumps(state.schema.model_dump(), indent=2)

            # Run structured generation for semantic search
            sql_generation_output = run_prompt(
                prompt_name="generate_semantic_review_sql",
                variables={
                    "schema_context": schema_json,
                    "question": state.question,
                    "query_vector": query_vector,
                    "threshold": self.semantic_search_threshold,
                },
                config=self.config,
                chat_history=state.chat_history,
                output_schema=SQLGenerator,
            )

            logger.info("Semantic SQL generation successful.")
            if sql_generation_output.thought_process:
                logger.info(
                    f"Thought Process snippet: {sql_generation_output.thought_process[:100]}..."
                )

            chat_update = []
            if sql_generation_output.sql_query:
                chat_update = [
                    AIMessage(
                        content=f"Generated Semantic SQL:\n```sql\n{sql_generation_output.sql_query}\n```"
                    )
                ]
            elif sql_generation_output.unsupported_explanation:
                chat_update = [
                    AIMessage(content=sql_generation_output.unsupported_explanation)
                ]

            # Ensure we reset the fuzzy warning if the new generation didn't provide one
            if not sql_generation_output.fuzzy_match_warning:
                sql_generation_output = sql_generation_output.model_copy(
                    update={"fuzzy_match_warning": None}
                )

            return {"generated_sql": sql_generation_output, "chat_history": chat_update}

        except Exception as e:
            logger.exception("Failed to create Semantic SQL query.")
            raise SQLGenerationError(e)

    def generate_sql(self, state: TextToSQLState):
        try:
            logger.info(
                f"Generating SQL query for question: '{state.question}' (Iteration: {state.validator.iteration_count + 1})"
            )

            # Serialize the schema so the LLM can read it
            schema_json = json.dumps(state.schema.model_dump(), indent=2)

            # Run structured generation
            sql_generation_output = run_prompt(
                prompt_name="generate_sql",
                variables={
                    "schema_context": schema_json,
                    "question": state.question,
                },
                config=self.config,
                chat_history=state.chat_history,
                output_schema=SQLGenerator,
            )

            logger.info("SQL generation successful.")
            logger.info(
                f"Thought Process snippet: {sql_generation_output.thought_process[:100]}..."
            )

            chat_update = []
            if sql_generation_output.sql_query:
                chat_update = [
                    AIMessage(
                        content=f"Generated SQL:\n```sql\n{sql_generation_output.sql_query}\n```"
                    )
                ]
            elif sql_generation_output.unsupported_explanation:
                chat_update = [
                    AIMessage(content=sql_generation_output.unsupported_explanation)
                ]

            # Ensure we reset the fuzzy warning if the new generation didn't provide one
            if not sql_generation_output.fuzzy_match_warning:
                sql_generation_output = sql_generation_output.model_copy(
                    update={"fuzzy_match_warning": None}
                )

            return {"generated_sql": sql_generation_output, "chat_history": chat_update}

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
            cleaned_sql = clean_sql_query(sql)

            # Update the SQL in our generated_sql update object
            generated_sql_update = state.generated_sql.model_copy(
                update={"sql_query": cleaned_sql}
            )

            # If the model explicitly said it can't answer, don't run EXPLAIN and mark as invalid
            if not cleaned_sql:
                logger.info(
                    "No SQL generated (unsupported question). Skipping EXPLAIN validation."
                )
                return {
                    "generated_sql": generated_sql_update,
                    "validator": state.validator.model_copy(
                        update={
                            "is_valid_query": True,  # Mark as valid so we don't retry, but execute will skip
                            "error_message": None,
                        }
                    ),
                }

            logger.info(
                f"Validating SQL via EXPLAIN (Iteration: {state.validator.iteration_count + 1})"
            )

            # Execute EXPLAIN in Snowflake
            with self.config.get_snowflake_connection(write_access=False) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"EXPLAIN {cleaned_sql}")

            logger.info("SQL validation successful.")
            return {
                "generated_sql": generated_sql_update,
                "validator": state.validator.model_copy(
                    update={
                        "is_valid_query": True,
                        "error_message": None,
                    }
                ),
            }

        except Exception as e:
            error_str = str(e)
            logger.warning(f"SQL validation failed: {error_str}")
            return {
                "validator": state.validator.model_copy(
                    update={
                        "is_valid_query": False,
                        "error_message": error_str,
                        "iteration_count": state.validator.iteration_count + 1,
                    }
                ),
                "chat_history": [
                    SystemMessage(
                        content=f"SQL Validation Error: {error_str}\nPlease correct the query based on this error."
                    )
                ],
            }

    def execute_sql(self, state: TextToSQLState):
        """
        Executes the validated SQL query against Snowflake.
        Fetches the results and stores them in the state.
        """
        try:
            if (
                not state.validator.is_valid_query
                or not state.generated_sql
                or not state.generated_sql.sql_query
            ):
                logger.info(
                    "Skipping execution: query is either invalid or unsupported."
                )
                return {"query_results": None}

            sql = state.generated_sql.sql_query
            logger.info("Executing SQL query in Snowflake...")

            with self.config.get_snowflake_connection(write_access=False) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    # Fetch one more than safety limit to detect truncation
                    rows = cursor.fetchmany(self.safety_limit + 1)
                    is_capped = len(rows) > self.safety_limit
                    # If capped, slice back to safety limit
                    results = process_snowflake_results(
                        cursor.description, rows[: self.safety_limit]
                    )

            if is_result_empty(results):
                error_str = "Query executed successfully but returned 0 results."
                logger.warning(error_str)
                return {
                    "validator": state.validator.model_copy(
                        update={
                            "is_valid_query": False,
                            "error_message": error_str,
                            "iteration_count": state.validator.iteration_count + 1,
                        }
                    ),
                    "chat_history": [SystemMessage(content=error_str)],
                }

            logger.info(f"Execution successful. Fetched {len(results)} rows.")
            return {"query_results": results, "is_capped": is_capped}

        except Exception as e:
            error_str = str(e)
            logger.warning(f"SQL execution failed: {error_str}")
            return {
                "validator": state.validator.model_copy(
                    update={
                        "is_valid_query": False,
                        "error_message": error_str,
                        "iteration_count": state.validator.iteration_count + 1,
                    }
                ),
                "chat_history": [
                    SystemMessage(
                        content=f"SQL Execution Error: {error_str}\nPlease correct the query based on this error."
                    )
                ],
            }

    def generate_tabular_answer(self, state: TextToSQLState):
        """
        Passes the query results to the tabular_answer state field, truncated to the display limit.
        """
        logger.info("Generating tabular answer...")
        limit = (
            self.semantic_tab_limit
            if state.router and state.router.is_semantic_intent
            else self.standard_tab_limit
        )

        total_count = len(state.query_results) if state.query_results else 0
        truncated_results = state.query_results[:limit] if state.query_results else []
        result_count = len(truncated_results)

        # Generate a transparency disclaimer for tabular results
        answer = "Here are the tabular results you requested."
        if total_count > result_count:
            answer = f"Showing the first {result_count} records out of {total_count} total matches found. The full dataset can be downloaded as a CSV below."

        if state.is_capped:
            sql_query = state.generated_sql.sql_query if state.generated_sql else "N/A"
            answer += (
                f"\n\n---\n**Note**: The full results exceeded our safety limit of {self.safety_limit} records. "
                f"Please run the following query directly in your database to get all requested records:\n"
                f"```sql\n{sql_query}\n```"
            )

        return {
            "tabular_answer": TabularResponse(
                data=truncated_results,
                answer=answer,
                total_count=total_count,
                is_capped=state.is_capped,
            ),
            "chat_history": [AIMessage(content=answer)],
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

            limit = self.standard_nl_limit
            total_count = len(state.query_results) if state.query_results else 0

            truncated_results = (
                state.query_results[:limit] if state.query_results else []
            )
            result_count = len(truncated_results)

            # Be mindful of result size. If it's too large, we might need to truncate.
            results_str = json.dumps(truncated_results, indent=2, default=str)
            if len(results_str) > 50000:
                results_str = results_str[:50000] + "\n... [TRUNCATED]"

            # Run NL generation
            response = run_prompt(
                prompt_name="generate_nl_answer",
                variables={
                    "question": state.question,
                    "sql_query_results": results_str,
                    "result_count": result_count,
                    "total_count": total_count,
                },
                config=self.config,
                chat_history=state.chat_history,
                use_fast_llm=True,
            )
            answer = response.content.strip()

            if state.is_capped:
                sql_query = (
                    state.generated_sql.sql_query if state.generated_sql else "N/A"
                )
                answer += (
                    f"\n\n---\n**Note**: The full results exceeded our safety limit of {self.safety_limit} records. "
                    f"Please run the following query directly in your database to get all requested records:\n"
                    f"```sql\n{sql_query}\n```"
                )

            return {"answer": answer, "chat_history": [AIMessage(content=answer)]}

        except Exception as e:
            logger.exception("Failed to generate natural language answer.")
            raise NodeException(f"Failed to generate NL answer: {e}")

    def summarize_review_sentiment(self, state: TextToSQLState):
        """
        Uses an LLM to generate a specialized natural language summary of review sentiments.
        """
        try:
            logger.info("Generating review sentiment summary...")

            # If the query was unsupported, just return the explanation as the answer
            if state.generated_sql and state.generated_sql.unsupported_explanation:
                explanation = state.generated_sql.unsupported_explanation
                return {
                    "answer": explanation,
                    "chat_history": [AIMessage(content=explanation)],
                }

            limit = self.semantic_nl_limit
            total_count = len(state.query_results) if state.query_results else 0

            truncated_results = (
                state.query_results[:limit] if state.query_results else []
            )
            result_count = len(truncated_results)

            results_str = json.dumps(truncated_results, indent=2, default=str)
            if len(results_str) > 50000:
                results_str = results_str[:50000] + "\n... [TRUNCATED]"

            # Run NL generation
            response = run_prompt(
                prompt_name="summarize_review_sentiment",
                variables={
                    "question": state.question,
                    "sql_query_results": results_str,
                    "result_count": result_count,
                    "total_count": total_count,
                },
                config=self.config,
                chat_history=state.chat_history,
                use_fast_llm=False,  # Use smart LLM for better thematic grouping
            )
            answer = response.content.strip()

            if state.is_capped:
                sql_query = (
                    state.generated_sql.sql_query if state.generated_sql else "N/A"
                )
                answer += (
                    f"\n\n---\n**Note**: The full results exceeded our safety limit of {self.safety_limit} records. "
                    f"Please run the following query directly in your database to get all requested records:\n"
                    f"```sql\n{sql_query}\n```"
                )

            return {"answer": answer, "chat_history": [AIMessage(content=answer)]}

        except Exception as e:
            logger.exception("Failed to generate review sentiment summary.")
            raise NodeException(f"Failed to generate review summary: {e}")
