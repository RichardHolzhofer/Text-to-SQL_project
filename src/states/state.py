from pydantic import BaseModel, Field
from typing_extensions import List, Literal


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
