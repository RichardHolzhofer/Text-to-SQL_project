import json
from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.config.config import Config
from src.database.db import SupabaseDB
from src.exceptions.exception import NodeException, SchemaBuildError, SQLGenerationError
from src.guardrails.guardrails import get_guardrails
from src.logger.logger import logger
from src.states.state import (
    ConversationEvaluator,
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
            self.semantic_search_threshold = 0.4
            self.semantic_tab_limit = 10
            self.semantic_nl_limit = 100
            self.standard_tab_limit = 10
            self.standard_nl_limit = 100
            self.mart_schema_path = "olist/models/marts/_marts_schema.yml"
            self.enhancement_schema_path = "embeddings/_embeddings_schema.yml"
            self.safety_limit = 1000
            self.review_language = "Portuguese"
            # Layer 1 Protection: Exclude sensitive tables and columns from the LLM schema
            self.exclude_tables = []
            self.exclude_columns = [
                "credit_card_amount",
                "boleto_amount",
                "voucher_amount",
                "customer_zip_code",
                "max_installments",
            ]
            logger.info("TextToSQLNodes initialized using Config LLMs and Supabase.")
        except Exception as e:
            logger.exception("Failed to initialize TextToSQLNodes.")
            raise NodeException(e)

    def input_guardrail_node(self, state: TextToSQLState, config: RunnableConfig):
        """
        Scans the user's question for PII and prompt injection before it hits the LLM.
        """
        logger.info("Node: Input Guardrail")
        guardrails = get_guardrails()

        # 1. Sanitize the current question
        sanitized_question = guardrails.scan_user_input(state.question)

        # Detect if it was flagged/blocked (contains standard safety text)
        is_blocked = "flagged for safety reasons" in sanitized_question

        if is_blocked:
            logger.warning(
                "Prompt injection / security threat flagged. Short-circuiting workflow."
            )
            return {
                "sanitized_question": sanitized_question,
                "answer": sanitized_question,  # Populate answer immediately
                "chat_history": [AIMessage(content=sanitized_question)],
            }

        # 2. Sanitize the chat history (messages from previous turns)
        # This prevents PII saved in the DB from leaking back to the LLM
        sanitized_history = guardrails.scan_history(state.chat_history)

        # Populate sanitized_question and sanitized_history
        return {
            "sanitized_question": sanitized_question,
            "chat_history": sanitized_history,
        }

    def conversation_evaluator_node(
        self, state: TextToSQLState, config: RunnableConfig
    ):
        """
        Determines if the user's question is a general conversation or a database query.
        """
        try:
            logger.info("Node: Conversation Evaluator")
            response = run_prompt(
                prompt_name="evaluate_conversation",
                variables={"question": state.sanitized_question},
                config=self.config,
                output_schema=ConversationEvaluator,
                use_fast_llm=True,
                runnable_config=config,
            )
            return {"is_general_conversation": response.is_general_conversation}
        except Exception:
            logger.exception("Failed to evaluate conversation intent.")
            # Default to false (SQL query) on error
            return {"is_general_conversation": False}

    def handle_general_conversation_node(
        self, state: TextToSQLState, config: RunnableConfig
    ):
        """
        Handles general conversations using a standard LLM.
        """
        try:
            logger.info("Node: Handle General Conversation")
            response = run_prompt(
                prompt_name="general_conversation",
                variables={"question": state.sanitized_question},
                config=self.config,
                chat_history=state.chat_history,
                use_fast_llm=True,
                runnable_config=config,
            )
            answer = response.content.strip()
            return {"answer": answer, "chat_history": [AIMessage(content=answer)]}
        except Exception:
            logger.exception("Failed to handle general conversation.")
            answer = "I'm sorry, I'm having trouble responding right now."
            return {"answer": answer, "chat_history": [AIMessage(content=answer)]}

    def build_schema(
        self,
        state: TextToSQLState,
        config: RunnableConfig,
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

            if state.db_schema is not None:
                logger.info("Schema is already loaded in state. Skipping extraction.")
                reset_state["db_schema"] = state.db_schema
                return reset_state

            # 2. Check Supabase Cache (Unless force_refresh is True)
            if not state.force_refresh:
                cache_data = self.db.get_schema_cache("unified_schema")
                if cache_data:
                    logger.info("Schema found in Supabase cache. Loading...")
                    try:
                        final_schema = Schema.model_validate(cache_data)
                        reset_state["db_schema"] = final_schema
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

            # Data Protection Filtering: Prune sensitive tables and columns
            filtered_models = []
            for model in unified_model_list:
                model_name = model.get("name")
                if model_name in self.exclude_tables:
                    logger.info(f"Layer 1: Filtering out entire table '{model_name}'")
                    continue

                if "columns" in model and self.exclude_columns:
                    original_cols = model["columns"]
                    model["columns"] = [
                        c
                        for c in original_cols
                        if c.get("name") not in self.exclude_columns
                    ]
                    removed_count = len(original_cols) - len(model["columns"])
                    if removed_count > 0:
                        logger.info(
                            f"Layer 1: Filtered {removed_count} sensitive columns from '{model_name}'"
                        )

                filtered_models.append(model)

            unified_model_list = filtered_models

            logger.info(
                f"Identified {len(unified_model_list)} models to extract after filtering."
            )

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
                    runnable_config=config,
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
                runnable_config=config,
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

            reset_state["db_schema"] = final_schema
            return reset_state

        except Exception as e:
            logger.exception("Failed to build unified schema.")
            raise SchemaBuildError(e)

    def route_format(self, state: TextToSQLState, config: RunnableConfig):
        """
        Uses an LLM to determine if the user wants tabular or natural language format,
        and also detects review and semantic intents.
        """
        try:
            # Run structured routing
            response = run_prompt(
                prompt_name="evaluate_intent",
                variables={"question": state.sanitized_question},
                config=self.config,
                output_schema=Router,
                use_fast_llm=True,
                runnable_config=config,
            )

            return {"router": response}

        except Exception:
            logger.exception("Failed to route format.")
            # Default fallback
            return {"router": Router(route="nl")}

    def extract_semantic_concept(self, state: TextToSQLState, config: RunnableConfig):
        """
        Extracts the search concept from the question and generates its embedding.
        """
        try:
            logger.info(
                f"Extracting search concept for question: '{state.sanitized_question}'"
            )

            # 1. Generate search concept for embedding
            concept_response = run_prompt(
                prompt_name="extract_search_concept",
                variables={
                    "question": state.sanitized_question,
                    "target_language": self.review_language,
                },
                config=self.config,
                use_fast_llm=True,
                runnable_config=config,
            )
            search_concept = concept_response.content.strip()
            logger.info(f"Extracted concept: '{search_concept}'")

            # 2. Generate embedding for the extracted concept
            logger.info(f"Generating embedding for concept: '{search_concept}'")
            embedding_model = self.config.get_embedding_model()
            query_vector = embedding_model.embed_query(search_concept)

            logger.info(f"Generated embedding of length {len(query_vector)}")

            return {
                "generated_sql": SQLGenerator(
                    search_concept=search_concept, query_vector=query_vector
                )
            }

        except Exception as e:
            logger.exception(
                "Failed to extract semantic concept or generate embedding."
            )
            raise NodeException(e)

    def semantic_query_generator(self, state: TextToSQLState, config: RunnableConfig):
        """
        Generates the specialized vector SQL using the semantic prompt.
        """
        try:
            logger.info(
                f"Generating Semantic SQL query for question: '{state.sanitized_question}'"
            )

            # Serialize the schema so the LLM can read it
            schema_json = json.dumps(state.db_schema.model_dump(), indent=2)

            # Run structured generation for semantic search
            sql_generation_output = run_prompt(
                prompt_name="generate_semantic_review_sql",
                variables={
                    "schema_context": schema_json,
                    "question": state.sanitized_question,
                    "threshold": self.semantic_search_threshold,
                },
                config=self.config,
                chat_history=state.chat_history,
                output_schema=SQLGenerator,
                runnable_config=config,
            )

            # Preserve the search concept and vector from the previous state (if any)
            if state.generated_sql:
                logger.info(
                    f"Preserving semantic context from state. Search concept: {state.generated_sql.search_concept}"
                )
                sql_generation_output = sql_generation_output.model_copy(
                    update={
                        "search_concept": state.generated_sql.search_concept
                        or sql_generation_output.search_concept,
                        "query_vector": state.generated_sql.query_vector
                        or sql_generation_output.query_vector,
                    }
                )
            else:
                logger.warning(
                    "No previous generated_sql found in state to preserve semantic context."
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

    def generate_sql(self, state: TextToSQLState, config: RunnableConfig):
        try:
            logger.info(
                f"Generating SQL query for question: '{state.sanitized_question}' (Iteration: {state.validator.iteration_count + 1})"
            )

            # Serialize the schema so the LLM can read it
            schema_json = json.dumps(state.db_schema.model_dump(), indent=2)

            # Run structured generation
            sql_generation_output = run_prompt(
                prompt_name="generate_sql",
                variables={
                    "schema_context": schema_json,
                    "question": state.sanitized_question,
                },
                config=self.config,
                chat_history=state.chat_history,
                output_schema=SQLGenerator,
                runnable_config=config,
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

    def validate_sql(self, state: TextToSQLState, config: RunnableConfig):
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

            # Inject the actual vector if the placeholder is present
            if "[QUERY_VECTOR]" in cleaned_sql:
                if not state.generated_sql or not state.generated_sql.query_vector:
                    msg = "SQL contains [QUERY_VECTOR] placeholder, but query_vector is missing from state.generated_sql."
                    logger.error(msg)
                    raise ValueError(msg)

                logger.info(
                    f"Injecting query_vector (length: {len(state.generated_sql.query_vector)}) into SQL for EXPLAIN."
                )
                cleaned_sql = cleaned_sql.replace(
                    "[QUERY_VECTOR]", str(state.generated_sql.query_vector)
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

    def execute_sql(self, state: TextToSQLState, config: RunnableConfig):
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

            # Inject the actual vector if the placeholder is present
            if "[QUERY_VECTOR]" in sql:
                if not state.generated_sql or not state.generated_sql.query_vector:
                    msg = "SQL contains [QUERY_VECTOR] placeholder, but query_vector is missing from state.generated_sql."
                    logger.error(msg)
                    raise ValueError(msg)

                logger.info(
                    f"Injecting query_vector (length: {len(state.generated_sql.query_vector)}) into SQL for execution."
                )
                sql = sql.replace(
                    "[QUERY_VECTOR]", str(state.generated_sql.query_vector)
                )

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

    def results_guardrail_node(self, state: TextToSQLState, config: RunnableConfig):
        """
        Anonymizes IDs in the query results so the LLM doesn't see raw PII.
        """
        logger.info("Node: Results Guardrail")
        if not state.query_results:
            return {"sanitized_query_results": None}

        guardrails = get_guardrails()
        sanitized = guardrails.scan_data(state.query_results)
        return {"sanitized_query_results": sanitized}

    def deanonymize_sql_node(self, state: TextToSQLState, config: RunnableConfig):
        """
        Restores real IDs in the generated SQL so it can be executed by the database.
        """
        logger.info("Node: Deanonymize SQL")
        if not state.generated_sql or not state.generated_sql.sql_query:
            return {}

        guardrails = get_guardrails()
        # We use the original question as context for the deanonymizer
        restored_sql = guardrails.scan_llm_output(
            state.sanitized_question, state.generated_sql.sql_query
        )

        return {
            "generated_sql": state.generated_sql.model_copy(
                update={"sql_query": restored_sql}
            )
        }

    def deanonymize_answer_node(self, state: TextToSQLState, config: RunnableConfig):
        """
        Restores real IDs in the natural language answer for the final user.
        """
        logger.info("Node: Deanonymize Answer")
        if not state.answer:
            return {}

        guardrails = get_guardrails()
        restored_answer = guardrails.scan_llm_output(
            state.sanitized_question, state.answer
        )

        return {"answer": restored_answer}

    def generate_tabular_answer(self, state: TextToSQLState, config: RunnableConfig):
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

        total_count_str = f"{total_count}+" if state.is_capped else str(total_count)

        # Generate a transparency disclaimer for tabular results
        answer = "Here are the tabular results you requested."
        if total_count > result_count or state.is_capped:
            answer = f"Showing the first {result_count} records out of {total_count_str} total matches found. The full dataset can be downloaded as a CSV below."

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

    def generate_nl_answer(self, state: TextToSQLState, config: RunnableConfig):
        """
        Uses an LLM to generate a natural language summary of the query results.
        """
        try:
            logger.info(
                f"Generating natural language answer for question: '{state.sanitized_question}'"
            )

            # If the query was unsupported, just return the explanation as the answer
            if state.generated_sql and state.generated_sql.unsupported_explanation:
                explanation = state.generated_sql.unsupported_explanation
                return {
                    "answer": explanation,
                    "chat_history": [AIMessage(content=explanation)],
                }

            limit = self.standard_nl_limit
            total_count = (
                len(state.sanitized_query_results)
                if state.sanitized_query_results
                else 0
            )

            truncated_results = (
                state.sanitized_query_results[:limit]
                if state.sanitized_query_results
                else []
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
                    "question": state.sanitized_question,
                    "sql_query_results": results_str,
                    "result_count": result_count,
                    "total_count": total_count,
                    "is_capped": state.is_capped,
                },
                config=self.config,
                chat_history=state.chat_history,
                use_fast_llm=False,
                runnable_config=config,
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

    def summarize_review_sentiment(self, state: TextToSQLState, config: RunnableConfig):
        """
        Uses an LLM to generate a specialized natural language summary of review sentiments.
        """
        try:
            logger.info(
                f"Generating review sentiment summary for question: '{state.sanitized_question}'"
            )

            # If the query was unsupported, just return the explanation as the answer
            if state.generated_sql and state.generated_sql.unsupported_explanation:
                explanation = state.generated_sql.unsupported_explanation
                return {
                    "answer": explanation,
                    "chat_history": [AIMessage(content=explanation)],
                }

            limit = self.semantic_nl_limit
            total_count = (
                len(state.sanitized_query_results)
                if state.sanitized_query_results
                else 0
            )

            truncated_results = (
                state.sanitized_query_results[:limit]
                if state.sanitized_query_results
                else []
            )
            result_count = len(truncated_results)

            results_str = json.dumps(truncated_results, indent=2, default=str)
            if len(results_str) > 50000:
                results_str = results_str[:50000] + "\n... [TRUNCATED]"

            # Run NL generation
            response = run_prompt(
                prompt_name="summarize_review_sentiment",
                variables={
                    "question": state.sanitized_question,
                    "sql_query_results": results_str,
                    "result_count": result_count,
                    "total_count": total_count,
                    "is_capped": state.is_capped,
                },
                config=self.config,
                chat_history=state.chat_history,
                use_fast_llm=False,  # Use smart LLM for better thematic grouping
                runnable_config=config,
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

    def persist_history(self, state: TextToSQLState, config: RunnableConfig):
        """
        Saves the conversation to the Supabase database.
        """
        try:
            # 1. Extract thread_id and user identity
            thread_id = config.get("configurable", {}).get("thread_id")
            # Prefer state values (passed from UI) but fallback to config
            user_id = state.user_id or config.get("configurable", {}).get("user_id")
            user_email = state.user_email or config.get("configurable", {}).get(
                "user_email"
            )

            if not thread_id or not user_id:
                logger.warning(
                    f"Missing thread_id ({thread_id}) or user_id ({user_id}). Skipping persistence."
                )
                return {}

            logger.info(f"Persisting history for thread {thread_id} and user {user_id}")

            # 2. Configure the database instance with user context
            self.db.user_id = user_id
            self.db.user_email = user_email

            # 3. Ensure Supabase chat history thread entry exists (upsert)
            # Use the sanitized question as title to prevent PII in thread titles
            self.db.upsert_chat_thread(
                thread_id, title=state.sanitized_question[:30] + "..."
            )

            # 4. Determine Assistant Message content and type
            content = None
            msg_type = "text"

            if state.tabular_answer:
                content = {
                    "data": state.tabular_answer.data,
                    "answer": state.tabular_answer.answer,
                }
                msg_type = "dataframe"
            elif state.answer:
                content = state.answer
                if "flagged for safety reasons" in state.answer:
                    msg_type = "warning"
                else:
                    msg_type = "text"
            elif state.generated_sql and state.generated_sql.unsupported_explanation:
                content = state.generated_sql.unsupported_explanation
                msg_type = "warning"

            # 5. Save messages to Supabase
            if content:
                # Save User message first
                self.db.save_message(thread_id, "user", state.question, "text")
                # Save Assistant message
                self.db.save_message(thread_id, "assistant", content, msg_type)
                logger.info(
                    "Successfully saved user and assistant messages to Supabase."
                )
            else:
                logger.warning("No assistant content found to save.")

            return {}
        except Exception as e:
            logger.error(f"Failed to persist history to Supabase: {e}")
            return {}
