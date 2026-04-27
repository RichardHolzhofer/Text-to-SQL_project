from typing import Annotated, Optional
from pydantic import BaseModel, Field
from typing_extensions import List, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

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


class RelationshipDigest(BaseModel):
    relationships: List[ForeignKeyRelationship] = Field(
        default_factory=list,
        description="Explicit foreign key relationships between tables",
    )


### State for SQL generator


class SQLGenerator(BaseModel):
    thought_process: str = Field(
        description="The 'Chain of Thought' reasoning before writing the SQL query."
    )
    sql_query: str = Field(
        description="The actual Snowflake SQL query produced by the generator."
    )


### TextToSQLState main class


class TextToSQLState(BaseModel):
    question: str
    chat_history: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list
    )

    # Schema extraction
    schema: Optional[Schema] = None

    # SQL Generation State
    generated_sql: Optional[SQLGenerator] = None

    # Validation / Looping State
    is_valid_query: bool = Field(
        default=False, description="Flag set by a 'Validator' node."
    )
    error_message: Optional[str] = Field(
        default=None, description="The raw error string from Snowflake."
    )
    iteration_count: int = Field(
        default=0, description="Counter for error-fixing loops."
    )

    answer: Optional[str] = Field(
        default=None, description="Final natural language answer."
    )
