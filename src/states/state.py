from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import List, Literal

### States for schema extraction


class ColumnTest(BaseModel):
    test_type: Literal[
        "not_null", "unique", "accepted_values", "relationships", "other"
    ] = Field(
        description="Type of dbt column test",
    )
    accepted_values: List[str] | None = Field(
        default=None,
        description="Accepted values when test_type is accepted_values",
    )
    reference_table: str | None = Field(
        default=None,
        description="Referenced table when test_type is relationships",
    )
    reference_column: str | None = Field(
        default=None,
        description="Referenced column when test_type is relationships",
    )


class Column(BaseModel):
    column_name: str = Field(description="Name of the column")
    description: str | None = Field(
        default=None,
        description="Column description copied from source metadata when available",
    )
    possible_synonyms: List[str] | None = Field(
        default=None,
        description="Possible business synonyms users might use for this column",
    )
    tests: List[ColumnTest] | None = Field(
        default=None,
        description="dbt tests defined for this column",
    )


class Table(BaseModel):
    table_name: str = Field(description="Table name")
    description: str | None = Field(
        default=None,
        description="Table description copied from source metadata when available",
    )
    grain: str | None = Field(
        default=None,
        description="Table grain when explicitly present in source metadata",
    )
    primary_key: List[str] | None = Field(
        default=None,
        description="Primary key columns explicitly supported by source metadata",
    )
    columns: List[Column] = Field(
        default_factory=list,
        description="Columns in the table",
    )


class ForeignKeyRelationship(BaseModel):
    source_table: str = Field(description="Table containing the foreign key")
    source_column: str = Field(description="Foreign key column in the source table")
    target_table: str = Field(description="Referenced table")
    target_column: str = Field(description="Referenced column in the target table")


class Schema(BaseModel):
    tables: List[Table] = Field(
        default_factory=list,
        description="Tables available to the text-to-sql agent",
    )
    relationships: List[ForeignKeyRelationship] = Field(
        default_factory=list,
        description="Explicit foreign key relationships between tables",
    )
    updated_at: Optional[str] = Field(
        default=None,
        description="The timestamp when this schema was last synced/built.",
    )


class RelationshipDigest(BaseModel):
    relationships: List[ForeignKeyRelationship] = Field(
        default_factory=list,
        description="Explicit foreign key relationships between tables",
    )


### State for Router


class ConversationEvaluator(BaseModel):
    is_general_conversation: bool = Field(
        default=False,
        description="True if the message is general conversation, greeting, or off-topic. False if it is related to databases, data analysis, or querying information.",
    )


class Router(BaseModel):
    route: Literal["nl", "tab"] = Field(
        default="nl",
        description="Determines the desired output format based on the user's question. Output 'tab' if the user asks for a list, table, detailed records, or multiple results (e.g., 'show me the top 5 customers'). Output 'nl' if the user asks a question expecting a direct answer, a count, an average, or a summary (e.g., 'who is the best customer?', 'how many orders did we have?').",
    )
    is_review_query: bool = Field(
        default=False,
        description="True if the user's question is asking about product reviews, customer feedback, opinions, or sentiments.",
    )
    is_semantic_intent: bool = Field(
        default=False,
        description="True if the user's question involves fuzzy concepts, feelings, sentiments, abstract features, or specific experiences that are unlikely to be exact matches in a database column (e.g., 'product quality', 'fast shipping', 'fabric softness', 'good battery life'). False for standard analytical questions (e.g., 'how many 5-star reviews?').",
    )
    use_semantic_search: bool = Field(
        default=False,
        description="Flag indicating if semantic vector search should be used.",
    )
    is_fallback: bool = Field(
        default=False,
        description="Flag indicating if we are currently in a fallback execution loop.",
    )


### State for SQL generator


class SQLGenerator(BaseModel):
    thought_process: Optional[str] = Field(
        default=None,
        description="The 'Chain of Thought' reasoning before writing the SQL query.",
    )
    sql_query: Optional[str] = Field(
        default=None,
        description="The actual Snowflake SQL query. Set to null if the question cannot be answered with the provided schema.",
    )
    unsupported_explanation: Optional[str] = Field(
        default=None,
        description="A polite explanation for the user if the question cannot be answered (e.g., missing tables or columns).",
    )
    fuzzy_match_warning: Optional[str] = Field(
        default=None,
        description="A brief warning explaining that EDITDISTANCE was used to find similar records (Standard SQL only). Do not use this for semantic/vector search limits.",
    )
    search_concept: Optional[str] = Field(
        default=None,
        description="The extracted core semantic concept used for vector search.",
    )
    query_vector: Optional[List[float]] = Field(
        default=None,
        description="The embedding vector for semantic search.",
    )


### State for Validator


class Validator(BaseModel):
    is_valid_query: Optional[bool] = Field(
        default=None,
        description="Flag set by a 'Validator' node. None means not validated yet.",
    )
    error_message: Optional[str] = Field(
        default=None, description="The raw error string from Snowflake."
    )
    iteration_count: int = Field(
        default=0, description="Counter for error-fixing loops."
    )


### State for Tabular Response


class TabularResponse(BaseModel):
    data: List[dict] = Field(
        default_factory=list, description="The truncated data for UI display."
    )
    answer: str = Field(
        default="", description="The transparency disclaimer or explanation."
    )
    total_count: int = Field(
        default=0, description="The total number of records found in the database."
    )
    is_capped: bool = Field(
        default=False,
        description="True if the results were capped by the safety limit.",
    )


### TextToSQLState main class


class TextToSQLState(BaseModel):
    question: str
    sanitized_question: Optional[str] = Field(
        default=None,
        description="Redacted version of the question for LLM consumption.",
    )
    user_id: Optional[str] = Field(
        default=None, description="The unique UUID of the authenticated user."
    )
    user_email: Optional[str] = Field(
        default=None, description="The email address of the authenticated user."
    )
    chat_history: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list
    )

    is_general_conversation: bool = Field(
        default=False,
        description="Flag indicating if the user's input is a general conversation rather than a database query.",
    )

    # Schema extraction
    db_schema: Optional[Schema] = None

    # Router / Intent State
    router: Optional[Router] = None

    # SQL Generation State
    generated_sql: Optional[SQLGenerator] = None

    # Validation / Looping State
    validator: Validator = Field(
        default_factory=Validator,
        description="State tracking for SQL validation and retry loops.",
    )

    # SQL Executer
    query_results: Optional[List[dict]] = Field(
        default=None, description="The results from executing the SQL query."
    )
    sanitized_query_results: Optional[List[dict]] = Field(
        default=None,
        description="Redacted version of query results for LLM consumption.",
    )
    is_capped: bool = Field(
        default=False,
        description="True if the results were capped by the safety limit (e.g., 5000 rows).",
    )

    # NL answer
    answer: Optional[str] = Field(
        default=None, description="Final natural language answer."
    )

    # Tabular answer
    tabular_answer: Optional[TabularResponse] = Field(
        default=None,
        description="The structured results if tabular format is requested.",
    )

    # Force Cache Refresh
    force_refresh: bool = Field(
        default=False,
        description="If True, the schema builder will ignore the cache and rebuild from scratch.",
    )
