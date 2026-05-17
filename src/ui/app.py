import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langgraph_sdk import get_sync_client

from src.config.config import Config
from src.database.db import SupabaseDB
from src.nodes.node import TextToSQLNodes
from src.states.state import TextToSQLState


@st.cache_data
def convert_to_csv(data):
    if not data:
        return ""
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode("utf-8")


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


# Initialize LangGraph Client
@st.cache_resource
def get_client():
    return get_sync_client(url=config.langgraph_url)


client = get_client()


# Initialize Nodes
@st.cache_resource
def get_nodes():
    return TextToSQLNodes(config=config)


nodes = get_nodes()


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
            # msg["content"] is now a dict with 'data' and 'answer' keys
            content = msg["content"]
            if isinstance(content, dict):
                if content.get("answer"):
                    st.markdown(content["answer"])
                df = pd.DataFrame(content.get("data", []))
            else:
                # Fallback for old history
                df = pd.DataFrame(content)
            st.dataframe(df)
        elif msg["type"] == "warning":
            st.warning(msg["content"])

        show_download = False
        if msg.get("full_data"):
            data_len = len(msg["full_data"])
            if msg["type"] == "dataframe":
                if data_len > 10:
                    show_download = True
            elif msg["type"] == "text":
                if data_len > 5:
                    show_download = True

        if show_download:
            csv = convert_to_csv(msg["full_data"])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            user_email = st.session_state.get("user_email", "user").split("@")[0]
            st.download_button(
                label="Download Full Results (CSV)",
                data=csv,
                file_name=f"{user_email}_{timestamp}.csv",
                mime="text/csv",
                key=f"download_full_hist_{i}",
            )

# Chat input
prompt = st.chat_input("Ask a question about your data...")
if prompt:
    # Reset the new session flag as it's now being used
    st.session_state.new_session_requested = False

    # Add user message to state and display
    st.session_state.messages.append(
        {"role": "user", "type": "text", "content": prompt}
    )
    # Note: db.save_message is now handled by the backend persistence node

    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
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

                metadata_dict = {
                    "schema_updated_at": schema_ts,
                    "fast_llm": config.fast_model,
                    "smart_llm": config.smart_model,
                    "langfuse_session_id": st.session_state.thread_id,
                    "langfuse_user_id": st.session_state.user_email,
                }

                # Langfuse tracing is handled in the LangGraph backend process.
                # Pass attributes through runnable config metadata so backend node
                # callbacks and nested LLM calls can inherit the same session.
                graph_config = {
                    "configurable": {"thread_id": st.session_state.thread_id},
                    "metadata": metadata_dict,
                    "run_name": "text-to-sql-app",
                }

                # Ensure thread exists in LangGraph
                client.threads.create(
                    thread_id=st.session_state.thread_id,
                    if_exists="do_nothing",
                )

                # Invoke graph via LangGraph API
                result = client.runs.wait(
                    st.session_state.thread_id,
                    "text_to_sql",
                    input={
                        "question": prompt,
                        "user_id": st.session_state.user_id,
                        "user_email": st.session_state.user_email,
                        "chat_history": [{"role": "user", "content": prompt}],
                    },
                    config=graph_config,
                    metadata=metadata_dict,
                )

                # Determine response type and display
                # Check for prompt injection safety block
                if (
                    result.get("answer")
                    and "flagged for safety reasons" in result["answer"]
                ):
                    answer = result["answer"]
                    st.warning(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "type": "warning", "content": answer}
                    )

                # Check for error/unsupported explanation
                elif result.get("generated_sql") and result["generated_sql"].get(
                    "unsupported_explanation"
                ):
                    explanation = result["generated_sql"]["unsupported_explanation"]
                    st.warning(explanation)
                    st.session_state.messages.append(
                        {"role": "assistant", "type": "warning", "content": explanation}
                    )

                # Check tabular intent
                elif (
                    result.get("router")
                    and result["router"].get("route") == "tab"
                    and result.get("tabular_answer") is not None
                ):
                    # Display fuzzy match warning if present
                    if result.get("generated_sql") and result["generated_sql"].get(
                        "fuzzy_match_warning"
                    ):
                        warning_msg = result["generated_sql"]["fuzzy_match_warning"]
                        st.warning(warning_msg)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "type": "warning",
                                "content": warning_msg,
                            }
                        )

                    tab_resp = result["tabular_answer"]
                    if tab_resp.get("answer"):
                        st.markdown(tab_resp["answer"])

                    data = tab_resp.get("data")
                    full_results = result.get("query_results")
                    df = pd.DataFrame(data)
                    st.dataframe(df)

                    if full_results and len(full_results) > 10:
                        csv = convert_to_csv(full_results)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        user_email = st.session_state.get("user_email", "user").split(
                            "@"
                        )[0]
                        st.download_button(
                            label="Download Full Results (CSV)",
                            data=csv,
                            file_name=f"{user_email}_{timestamp}.csv",
                            mime="text/csv",
                            key=f"download_full_new_tab_{len(st.session_state.messages)}",
                        )

                    # Store combined data in content for history
                    combined_content = {
                        "data": data,
                        "answer": tab_resp.get("answer"),
                    }
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "type": "dataframe",
                            "content": combined_content,
                            "full_data": full_results,
                        }
                    )

                # Fallback to Natural Language
                elif result.get("answer"):
                    # Display fuzzy match warning if present
                    if result.get("generated_sql") and result["generated_sql"].get(
                        "fuzzy_match_warning"
                    ):
                        warning_msg = result["generated_sql"]["fuzzy_match_warning"]
                        st.warning(warning_msg)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "type": "warning",
                                "content": warning_msg,
                            }
                        )

                    answer = result["answer"]
                    full_results = result.get("query_results")
                    st.markdown(answer)

                    if full_results and len(full_results) > 5:
                        csv = convert_to_csv(full_results)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        user_email = st.session_state.get("user_email", "user").split(
                            "@"
                        )[0]
                        st.download_button(
                            label="Download Full Results (CSV)",
                            data=csv,
                            file_name=f"{user_email}_{timestamp}.csv",
                            mime="text/csv",
                            key=f"download_full_new_nl_{len(st.session_state.messages)}",
                        )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "type": "text",
                            "content": answer,
                            "full_data": full_results,
                        }
                    )
                else:
                    st.error(
                        "An unexpected error occurred. No answer or table was generated."
                    )

                # --- No need to generate title here as backend creates thread with title ---

            except Exception as e:
                error_msg = f"Error invoking graph: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "type": "warning", "content": error_msg}
                )
            finally:
                # Ensure all traces are flushed to Langfuse before Streamlit exits or reruns
                config.get_langfuse().flush()
