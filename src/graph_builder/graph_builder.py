from src.config.config import Config
from src.nodes.node import TextToSQLNodes
from src.states.state import TextToSQLState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
import json


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
        workflow.add_node("router_query_node", self.nodes.route_format)
        workflow.add_node("query_generator_node", self.nodes.generate_sql)
        workflow.add_node("validator_node", self.nodes.validate_sql)
        workflow.add_node("query_executer_node", self.nodes.execute_sql)
        workflow.add_node("generate_tabular_answer", self.nodes.generate_tabular_answer)
        workflow.add_node("generate_nl_answer", self.nodes.generate_nl_answer)

        # Edges
        workflow.set_entry_point("schema_builder_node")
        workflow.add_edge("schema_builder_node", "router_query_node")
        workflow.add_edge("router_query_node", "query_generator_node")
        workflow.add_edge("query_generator_node", "validator_node")
        workflow.add_edge("validator_node", "query_executer_node")

        # Routing to display format
        def format_router(state: TextToSQLState):
            if state.intent == "tab":
                return "tabular"
            return "nl"

        workflow.add_conditional_edges(
            "query_executer_node",
            format_router,
            {
                "tabular": "generate_tabular_answer",
                "nl": "generate_nl_answer",
            },
        )

        workflow.add_edge("generate_tabular_answer", END)
        workflow.add_edge("generate_nl_answer", END)

        """
        # Routing logic
        def should_continue(state: TextToSQLState):
            # If the LLM explicitly said it can't answer, we stop (later: route to clarification)
            if state.generated_sql and state.generated_sql.unsupported_explanation:
                return "end"

            if state.is_valid_query:
                return "end"
            if state.iteration_count >= 3:
                return "end"
            return "retry"

        workflow.add_conditional_edges(
            "validator_node",
            should_continue,
            {
                "retry": "query_generator_node",
                "end": END,
            },
        )
        """
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
        {"question": "Give me the top 3 sellers by the number of products they sold"},
        config=config,
    )

    # Save the full state to a JSON file for inspection
    output_file = "logs/last_graph_state.json"

    # Pre-process the result to ensure Pydantic models (like Schema and SQLGenerator)
    # are converted to dicts instead of being saved as strings.
    serializable_result = {
        k: (v.model_dump() if hasattr(v, "model_dump") else v)
        for k, v in result.items()
    }

    with open(output_file, "w", encoding="utf-8") as f:
        # Use default=str only as a fallback for things like Datetime or LangChain messages
        json.dump(serializable_result, f, indent=2, default=str)

    print(f"\n--- FULL STATE SAVED TO {output_file} ---")
