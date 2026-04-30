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
        workflow.add_node("router_query_node", self.nodes.route_format)
        workflow.add_node("query_generator_node", self.nodes.generate_sql)
        workflow.add_node("validator_node", self.nodes.validate_sql)
        workflow.add_node("query_executer_node", self.nodes.execute_sql)
        workflow.add_node(
            "generate_tabular_answer_node", self.nodes.generate_tabular_answer
        )
        workflow.add_node("generate_nl_answer_node", self.nodes.generate_nl_answer)

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
                "tabular": "generate_tabular_answer_node",
                "nl": "generate_nl_answer_node",
            },
        )

        workflow.add_edge("generate_tabular_answer_node", END)
        workflow.add_edge("generate_nl_answer_node", END)

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
