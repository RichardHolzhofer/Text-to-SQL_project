from src.graph_builder.graph_builder import TextToSQLGraph

# Instantiate the graph builder
builder = TextToSQLGraph()

# Build and compile the graph
# This 'graph' variable will be the entry point for LangGraph Platform
graph = builder.build_graph()
