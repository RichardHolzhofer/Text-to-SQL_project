import json
from langchain.chat_models import init_chat_model

from src.states.state import Table, RelationshipDigest, Schema
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
                messages = load_prompt(
                    "src/prompts/extract_table.yaml",
                    {"model_yaml": model_yaml_str},
                )

                table_schema = self.table_extractor.invoke(messages)
                db_tables.append(table_schema)

            logger.info(f"Successfully extracted {len(db_tables)} table definitions.")

            # Extract relationships using the unified context
            logger.info("Starting relationship extraction across all tables...")
            relationship_context = {
                "source_yaml": mart_schema_yaml,
                "tables": [table.model_dump() for table in db_tables],
            }

            rel_messages = load_prompt(
                "src/prompts/extract_relationships.yaml",
                {"relationship_context": json.dumps(relationship_context, indent=2)},
            )

            relationship_digest = self.relationship_extractor.invoke(rel_messages)

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
            pass
        except Exception as e:
            logger.exception("Failed to create SQL query.")
            raise SQLGenerationError(e)
