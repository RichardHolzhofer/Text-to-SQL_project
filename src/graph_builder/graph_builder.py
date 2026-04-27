from src.config.config import Config
from src.nodes.node import TextToSQLNodes
from src.states.state import TextToSQLState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END


class TextToSQLGraph:
    def __init__(self):
        # Pass an instance of Config, not the class
        self.config = Config()
        self.nodes = TextToSQLNodes(config=self.config)
        self.graph = None
        self.memory = MemorySaver()

    def build_graph(self):
        # Define the StateGraph with the state schema
        workflow = StateGraph(TextToSQLState)

        # Nodes
        workflow.add_node("schema_builder_node", self.nodes.build_schema)
        workflow.add_node("query_generator_node", self.nodes.generate_sql)

        # Edges
        workflow.set_entry_point("schema_builder_node")
        workflow.add_edge("schema_builder_node", "query_generator_node")
        workflow.add_edge("query_generator_node", END)

        # Compile and add checkpointer for memory
        self.graph = workflow.compile(checkpointer=self.memory)

        return self.graph


if __name__ == "__main__":
    # Test script
    builder = TextToSQLGraph()
    compiled_graph = builder.build_graph()

    # We need a thread_id for the checkpointer
    config = {"configurable": {"thread_id": "test_thread"}}

    # Invoke the graph
    result = compiled_graph.invoke(
        {"question": "How many customers are there?"}, config=config
    )

    print("\n--- GRAPH OUTPUT ---")
    print(f"Question: {result['question']}")
    if result["schema"]:
        print(f"Schema extracted with {len(result['schema'].tables)} tables.")
        for table in result["schema"].tables:
            print(f" - {table.table_name}")
    else:
        print("Schema not found.")

    print(result["generated_sql"])
