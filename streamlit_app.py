import streamlit as st
import uuid
import pandas as pd
import json
from src.graph_builder.graph_builder import TextToSQLGraph
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage

langfuse = get_client()

# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(page_title="Text-to-SQL Agent", layout="wide")
st.title("Text-to-SQL Agent")

# Initialize Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# Initialize Graph
@st.cache_resource
def get_graph():
    builder = TextToSQLGraph()
    return builder.build_graph()


try:
    graph = get_graph()
except Exception as e:
    st.error(f"Failed to initialize the graph: {str(e)}")
    st.stop()

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    if st.button("Start New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("*(Future Feature: Load past conversations from database)*")

# Display chat messages from history
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown(
            "Hello! I am your Text-to-SQL Assistant. Ask me a question about your database!"
        )

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["type"] == "text":
            st.markdown(msg["content"])
        elif msg["type"] == "dataframe":
            df = pd.DataFrame(msg["content"])
            st.dataframe(df)
            if len(df) > 30:
                json_data = json.dumps(msg["content"], indent=2)
                st.download_button(
                    label="Download as JSON",
                    data=json_data,
                    file_name="results.json",
                    mime="application/json",
                    key=f"download_hist_{i}",
                )
        elif msg["type"] == "warning":
            st.warning(msg["content"])

# Chat input
prompt = st.chat_input("Ask a question about your data...")
if prompt:
    # Add user message to state and display
    st.session_state.messages.append(
        {"role": "user", "type": "text", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            try:
                langfuse_handler = CallbackHandler()
                config["callbacks"] = [langfuse_handler]

                # Graph state updates inside the propagate_attributes context
                with propagate_attributes(
                    trace_name="text-to-sql-app",
                    session_id=st.session_state.thread_id,
                    user_id="streamlit-user",
                ):
                    result = graph.invoke(
                        {
                            "question": prompt,
                            "chat_history": [HumanMessage(content=prompt)],
                        },
                        config=config,
                    )

                # Determine response type and display
                # Check for error/unsupported explanation
                if (
                    result.get("generated_sql")
                    and result["generated_sql"].unsupported_explanation
                ):
                    explanation = result["generated_sql"].unsupported_explanation
                    st.warning(explanation)
                    st.session_state.messages.append(
                        {"role": "assistant", "type": "warning", "content": explanation}
                    )

                # Check tabular intent
                elif result.get("intent") == "tab" and result.get("tabular_answer"):
                    data = result["tabular_answer"]
                    df = pd.DataFrame(data)
                    st.dataframe(df)

                    # Render download button if > 30 rows
                    if len(df) > 30:
                        json_data = json.dumps(data, indent=2)
                        st.download_button(
                            label="Download as JSON",
                            data=json_data,
                            file_name="results.json",
                            mime="application/json",
                            key=f"download_new_{len(st.session_state.messages)}",
                        )
                    st.session_state.messages.append(
                        {"role": "assistant", "type": "dataframe", "content": data}
                    )

                # Fallback to Natural Language
                elif result.get("answer"):
                    answer = result["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "type": "text", "content": answer}
                    )
                else:
                    st.error(
                        "An unexpected error occurred. No answer or table was generated."
                    )

            except Exception as e:
                error_msg = f"Error invoking graph: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "type": "warning", "content": error_msg}
                )
