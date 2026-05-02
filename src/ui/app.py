import streamlit as st
import uuid
import pandas as pd
import json
from datetime import datetime
from src.config.config import Config
from src.database.db import SupabaseDB
from src.graph_builder.graph_builder import TextToSQLGraph
from dotenv import load_dotenv
from langfuse import propagate_attributes
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage
from src.utils.llm_utils import generate_conversation_title
from src.states.state import TextToSQLState


# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(page_title="Text-to-SQL Agent", layout="wide")
st.title("Text-to-SQL Agent")


# Initialize Config and Database
@st.cache_resource
def get_db():
    config = Config()
    # Initialize Langfuse via Config to set up the singleton for the session
    config.get_langfuse()
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
    return builder.build_graph(), builder.nodes


try:
    graph, nodes = get_graph()
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
        if st.button("Logout", use_container_width=True):
            db.sign_out()
            st.session_state.user_id = None
            st.session_state.user_email = None
            st.session_state.messages = []
            st.rerun()

        with st.expander("Danger Zone"):
            if st.button("Delete Account", type="primary", use_container_width=True):
                if st.session_state.get("confirm_delete_account"):
                    # Use an admin instance for deletion
                    admin_db = SupabaseDB(config, admin=True)
                    if admin_db.delete_user(st.session_state.user_id):
                        st.success("Account deleted.")
                        st.session_state.user_id = None
                        st.session_state.user_email = None
                        st.session_state.messages = []
                        st.session_state.confirm_delete_account = False
                        st.rerun()
                    else:
                        st.error("Failed to delete account.")
                else:
                    st.session_state.confirm_delete_account = True
                    st.warning(
                        "Are you sure? This will delete ALL your data. Click again to confirm."
                    )

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
    threads = db.get_threads()
    if threads:
        latest_thread_id = threads[0]["thread_id"]
        st.session_state.thread_id = latest_thread_id
        st.session_state.messages = db.load_chat_history(latest_thread_id)
        st.rerun()

# --- Sidebar: Controls ---
with st.sidebar:
    st.header("History")
    threads = db.get_threads()
    if threads:
        for t in threads:
            title = t.get("title") or t["thread_id"]
            # Highlight the active thread by wrapping it in brackets or just bolding
            button_label = (
                f"💬 **{title}**"
                if t["thread_id"] == st.session_state.thread_id
                else title
            )

            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button(
                    button_label,
                    key=f"thread_{t['thread_id']}",
                    use_container_width=True,
                ):
                    st.session_state.thread_id = t["thread_id"]
                    st.session_state.messages = db.load_chat_history(t["thread_id"])
                    st.session_state.new_session_requested = (
                        False  # Reset flag when switching
                    )
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{t['thread_id']}", help="Delete thread"):
                    if db.delete_thread(t["thread_id"]):
                        if st.session_state.thread_id == t["thread_id"]:
                            st.session_state.thread_id = str(uuid.uuid4())
                            st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error("Delete failed")
    else:
        st.info("No past conversations found.")

    if st.button("Start New Conversation", type="primary", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.new_session_requested = True
        st.rerun()

    st.markdown("---")
    st.header("System")

    # Fetch the latest sync timestamp for the UI
    try:
        sync_response = (
            db.supabase_conn.table("schema_cache")
            .select("updated_at")
            .eq("key", "unified_schema")
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        last_sync_raw = (
            sync_response.data[0]["updated_at"] if sync_response.data else None
        )
        # Format the timestamp for better readability if it exists
        if last_sync_raw:
            # Parse UTC and convert to local system timezone
            dt_utc = datetime.fromisoformat(last_sync_raw.replace("Z", "+00:00"))
            dt_local = dt_utc.astimezone()
            last_sync_display = dt_local.strftime("%Y-%m-%d %H:%M:%S")
        else:
            last_sync_display = "Never"
    except Exception:
        last_sync_display = "Error fetching"

    if st.button(
        "🔄 Sync Schema Metadata",
        help="Rebuilds the schema cache from Snowflake/dbt (takes ~2 mins)",
        use_container_width=True,
    ):
        with st.status("Syncing metadata...", expanded=True) as status:
            try:
                st.write("Building unified schema...")
                # Create a dummy state with force_refresh=True
                dummy_state = TextToSQLState(
                    question="internal_sync", force_refresh=True
                )
                # Call the node directly
                nodes.build_schema(dummy_state)
                status.update(label="Sync Complete!", state="complete", expanded=False)
                st.success("Schema metadata updated successfully!")
                st.rerun()  # Rerun to refresh the timestamp display
            except Exception as e:
                status.update(label="Sync Failed", state="error")
                st.error(f"Sync failed: {e}")

    st.caption(f"Last Schema Sync: **{last_sync_display}**")

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

    # Ensure thread exists in DB before saving messages
    if len(st.session_state.messages) == 0:
        db.create_thread(st.session_state.thread_id, title=prompt[:30] + "...")

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
            graph_config = {"configurable": {"thread_id": st.session_state.thread_id}}

            try:
                langfuse_handler = CallbackHandler()
                graph_config["callbacks"] = [langfuse_handler]

                # Fetch the LATEST schema timestamp for tracing observability
                schema_info = (
                    db.supabase_conn.table("schema_cache")
                    .select("updated_at")
                    .eq("key", "unified_schema")
                    .order("updated_at", desc=True)
                    .limit(1)
                    .execute()
                )
                schema_ts = (
                    schema_info.data[0]["updated_at"]
                    if schema_info.data
                    else "Not Synced"
                )

                # Graph state updates inside the propagate_attributes context
                with propagate_attributes(
                    trace_name="text-to-sql-app",
                    session_id=st.session_state.thread_id,
                    user_id=st.session_state.user_email,
                    metadata={"schema_updated_at": schema_ts},
                ):
                    result = graph.invoke(
                        {
                            "question": prompt,
                            "chat_history": [HumanMessage(content=prompt)],
                        },
                        config=graph_config,
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

                # --- Auto-generate title after the first exchange ---
                if len(st.session_state.messages) == 2:
                    with st.spinner("Generating conversation title..."):
                        new_title = generate_conversation_title(config, prompt)
                        db.create_thread(st.session_state.thread_id, title=new_title)
                    st.rerun()

            except Exception as e:
                error_msg = f"Error invoking graph: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "type": "warning", "content": error_msg}
                )
