from src.config.config import Config
from src.nodes.node import TextToSQLNodes
from src.states.state import TextToSQLState
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langgraph.graph import StateGraph, END


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

    def build_graph(self):
        # Define the StateGraph with the state schema
        workflow = StateGraph(TextToSQLState)

        # Nodes
        workflow.add_node("schema_builder_node", self.nodes.build_schema)
        workflow.add_node("router_query_node", self.nodes.route_format)
        workflow.add_node("review_intent_node", self.nodes.review_intent_node)
        workflow.add_node("review_search_node", self.nodes.review_search_node)
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

        def after_router(state: TextToSQLState):
            if state.router and state.router.is_review_query:
                return "review"
            return "general"

        workflow.add_conditional_edges(
            "router_query_node",
            after_router,
            {
                "review": "review_intent_node",
                "general": "query_generator_node",
            },
        )

        def after_review_intent(state: TextToSQLState):
            if state.router and state.router.use_semantic_search:
                return "semantic"
            return "standard"

        workflow.add_conditional_edges(
            "review_intent_node",
            after_review_intent,
            {
                "semantic": "review_search_node",
                "standard": "query_generator_node",
            },
        )

        workflow.add_edge("review_search_node", "validator_node")
        workflow.add_edge("query_generator_node", "validator_node")
        workflow.add_edge("validator_node", "query_executer_node")

        # Routing to display format or fallback
        def format_router(state: TextToSQLState):
            # Check for fallback: 0 results on standard review query
            if (
                state.router
                and state.router.is_review_query
                and state.query_results is not None
                and len(state.query_results) == 0
                and not state.router.is_fallback
                and not state.router.use_semantic_search
            ):
                return "fallback"

            if state.router and state.router.route == "tab":
                return "tabular"
            return "nl"

        workflow.add_conditional_edges(
            "query_executer_node",
            format_router,
            {
                "fallback": "review_intent_node",
                "tabular": "generate_tabular_answer_node",
                "nl": "generate_nl_answer_node",
            },
        )

        workflow.add_edge("generate_tabular_answer_node", END)
        workflow.add_edge("generate_nl_answer_node", END)

        # Compile and add checkpointer for memory
        self.graph = workflow.compile(checkpointer=self.memory)

        return self.graph
