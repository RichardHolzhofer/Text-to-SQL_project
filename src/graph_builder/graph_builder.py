import os

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph
from psycopg_pool import ConnectionPool

from src.config.config import Config
from src.nodes.node import TextToSQLNodes
from src.states.state import TextToSQLState


class TextToSQLGraph:
    def __init__(self):
        # Pass an instance of Config, not the class
        self.config = Config()
        self.nodes = TextToSQLNodes(config=self.config)
        self.graph = None

        # Initialize Postgres connection pool for persistent memory
        self.pool = ConnectionPool(
            conninfo=self.config.sb_db_uri, max_size=20, kwargs={"autocommit": True}
        )
        self.memory = PostgresSaver(self.pool)

        # Ensure the checkpoint tables exist in Supabase
        self.memory.setup()

    def _route_after_intent(self, state: TextToSQLState):
        # Only route to semantic if it's a review query AND has fuzzy/semantic intent
        if (
            state.router
            and state.router.is_review_query
            and state.router.is_semantic_intent
        ):
            return "semantic"
        return "general"

    def _route_after_validator(self, state: TextToSQLState):
        if state.validator and not state.validator.is_valid_query:
            iteration = state.validator.iteration_count
            if iteration < self.nodes.max_retry:
                if state.router and state.router.is_review_query and iteration >= 2:
                    return "semantic_generator"
                return "standard_generator"
        return "execute"

    def _route_after_execute(self, state: TextToSQLState):
        if state.validator and not state.validator.is_valid_query:
            iteration = state.validator.iteration_count
            if iteration < self.nodes.max_retry:
                if state.router and state.router.is_review_query and iteration >= 2:
                    return "semantic_generator"
                return "standard_generator"

        if (
            state.router
            and state.router.is_review_query
            and state.router.route == "nl"
            and state.router.is_semantic_intent
        ):
            return "review_nl"

        if state.router and state.router.route == "tab":
            return "tabular"
        return "nl"

    def build_graph(self):
        # Define the StateGraph with the state schema
        workflow = StateGraph(TextToSQLState)

        # Nodes
        workflow.add_node("schema_builder_node", self.nodes.build_schema)
        workflow.add_node("router_query_node", self.nodes.route_format)
        workflow.add_node("query_generator_node", self.nodes.generate_sql)

        workflow.add_node(
            "extract_semantic_concept_node", self.nodes.extract_semantic_concept
        )
        workflow.add_node(
            "semantic_query_generator_node", self.nodes.semantic_query_generator
        )
        workflow.add_node("validator_node", self.nodes.validate_sql)
        workflow.add_node("query_executer_node", self.nodes.execute_sql)
        workflow.add_node(
            "generate_tabular_answer_node", self.nodes.generate_tabular_answer
        )
        workflow.add_node("generate_nl_answer_node", self.nodes.generate_nl_answer)
        workflow.add_node(
            "summarize_review_sentiment_node", self.nodes.summarize_review_sentiment
        )

        # Edges
        workflow.set_entry_point("schema_builder_node")
        workflow.add_edge("schema_builder_node", "router_query_node")

        workflow.add_conditional_edges(
            "router_query_node",
            self._route_after_intent,
            {
                "semantic": "extract_semantic_concept_node",
                "general": "query_generator_node",
            },
        )

        workflow.add_edge(
            "extract_semantic_concept_node", "semantic_query_generator_node"
        )
        workflow.add_edge("query_generator_node", "validator_node")
        workflow.add_edge("semantic_query_generator_node", "validator_node")

        workflow.add_conditional_edges(
            "validator_node",
            self._route_after_validator,
            {
                "standard_generator": "query_generator_node",
                "semantic_generator": "semantic_query_generator_node",
                "execute": "query_executer_node",
            },
        )

        workflow.add_conditional_edges(
            "query_executer_node",
            self._route_after_execute,
            {
                "standard_generator": "query_generator_node",
                "semantic_generator": "semantic_query_generator_node",
                "tabular": "generate_tabular_answer_node",
                "nl": "generate_nl_answer_node",
                "review_nl": "summarize_review_sentiment_node",
            },
        )

        workflow.add_edge("generate_tabular_answer_node", END)
        workflow.add_edge("generate_nl_answer_node", END)
        workflow.add_edge("summarize_review_sentiment_node", END)

        # Compile and add checkpointer for memory
        self.graph = workflow.compile(checkpointer=self.memory)

        return self.graph


if __name__ == "__main__":
    graph = TextToSQLGraph()

    compiled_graph = graph.build_graph()

    graph_image = compiled_graph.get_graph().draw_mermaid_png()

    os.makedirs("./graph_image", exist_ok=True)
    with open("./graph_image/graph_image.png", "wb") as f:
        f.write(graph_image)
