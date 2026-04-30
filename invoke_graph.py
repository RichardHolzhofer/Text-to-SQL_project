from src.graph_builder.graph_builder import TextToSQLGraph
import uuid
from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()


@observe()
def langgraph_pipeline(session_id, user_id, thread_id):
    with propagate_attributes(
        trace_name="text-to-sql-app", session_id=session_id, user_id=user_id
    ):
        langfuse_handler = CallbackHandler()

        # Test script
        builder = TextToSQLGraph()
        compiled_graph = builder.build_graph().with_config(callbacks=[langfuse_handler])

        # We need a thread_id for the checkpointer
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # Invoke the graph
        result = compiled_graph.invoke(
            {
                "question": "Give me the top 3 customers by the number of products they bought"
            },
            config=config,
        )

        return result


if __name__ == "__main__":
    session_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())

    langgraph_pipeline(session_id=session_id, user_id=user_id, thread_id=thread_id)
