import streamlit as st
import uuid
import pandas as pd
import json
from src.config.config import Config
from src.database.db import SupabaseDB
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


# Initialize Config and Database
@st.cache_resource
def get_db():
    config = Config()
    return SupabaseDB(config), config


db, config = get_db()

# Initialize Session State
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "new_session_requested" not in st.session_state:
    st.session_state.new_session_requested = False

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


# --- Sidebar: Authentication ---
with st.sidebar:
    st.header("Account")
    if st.session_state.user_id is None:
        auth_mode = st.radio("Mode", ["Login", "Sign Up"], horizontal=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if auth_mode == "Login":
            if st.button("Login"):
                success, message = db.sign_in(email, password)
                if success:
                    st.session_state.user_id = db.user_id
                    st.session_state.user_email = db.user_email
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        else:
            if st.button("Sign Up"):
                success, message = db.sign_up(email, password)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    else:
        st.write(f"Logged in as: **{st.session_state.user_email}**")
        if st.button("Logout"):
            db.sign_out()
            st.session_state.user_id = None
            st.session_state.user_email = None
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

# Stop the app here if user is not logged in
if st.session_state.user_id is None:
    st.info("Please login or sign up to use the Text-to-SQL Agent.")
    st.stop()

# Ensure the database instance knows who the current user is (for session-based calls)
db.user_id = st.session_state.user_id
db.user_email = st.session_state.user_email

# --- Auto-load last session if starting fresh (and not explicitly requesting a new one) ---
if (
    st.session_state.user_id
    and not st.session_state.messages
    and not st.session_state.new_session_requested
):
    sessions = db.get_sessions()
    if sessions:
        latest_session_id = sessions[0]["id"]
        st.session_state.thread_id = latest_session_id
        st.session_state.messages = db.load_chat_history(latest_session_id)
        st.rerun()

# --- Sidebar: Controls ---
with st.sidebar:
    st.header("History")
    sessions = db.get_sessions()
    if sessions:
        for s in sessions:
            title = s.get("title") or s["id"]
            # Use a slightly different label or style for the active session
            button_label = (
                f"💬 {title}" if s["id"] == st.session_state.thread_id else title
            )

            if st.button(
                button_label, key=f"session_{s['id']}", use_container_width=True
            ):
                st.session_state.thread_id = s["id"]
                st.session_state.messages = db.load_chat_history(s["id"])
                st.session_state.new_session_requested = (
                    False  # Reset flag when switching
                )
                st.rerun()
    else:
        st.info("No past conversations found.")

    st.markdown("---")
    if st.button("Start New Conversation", type="primary", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.new_session_requested = True
        st.rerun()

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
    # Reset the new session flag as it's now being used
    st.session_state.new_session_requested = False

    # Ensure session exists in DB before saving messages
    if len(st.session_state.messages) == 0:
        db.create_session(st.session_state.thread_id, title=prompt[:30] + "...")

    # Add user message to state and display
    st.session_state.messages.append(
        {"role": "user", "type": "text", "content": prompt}
    )
    db.save_message(st.session_state.thread_id, "user", prompt, "text")

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
                    db.save_message(
                        st.session_state.thread_id, "assistant", explanation, "warning"
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
                    db.save_message(
                        st.session_state.thread_id, "assistant", data, "dataframe"
                    )

                # Fallback to Natural Language
                elif result.get("answer"):
                    answer = result["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "type": "text", "content": answer}
                    )
                    db.save_message(
                        st.session_state.thread_id, "assistant", answer, "text"
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
