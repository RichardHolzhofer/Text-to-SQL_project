import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langgraph_sdk import get_sync_client

from src.config.config import Config
from src.database.db import SupabaseDB
from src.exceptions.exception import UIComponentLoadError, YAMLProcessingError
from src.nodes.node import TextToSQLNodes
from src.states.state import TextToSQLState
from src.utils.ui_utils import generate_chat_title, load_and_filter_schema


@st.cache_data
def convert_to_csv(data):
    if not data:
        return ""
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode("utf-8")


# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(page_title="QueryGraph", layout="wide")

try:
    with open("src/ui/styles/main.css", "r", encoding="utf-8") as f:
        CUSTOM_CSS = f"<style>\n{f.read()}\n</style>"
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
except Exception as e:
    st.error(str(UIComponentLoadError(e)))


# Initialize Config and Database
@st.cache_resource
def get_db():
    config = Config()
    # Initialize Langfuse via Config
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

if "thread_titles" not in st.session_state:
    st.session_state.thread_titles = {}


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


# Authentication & Landing Page
if st.session_state.user_id is None:
    st.markdown(
        '<h1 class="landing-title">Query<span class="accent-text">Graph</span></h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="landing-subtitle">Bridging the gap between you and your database</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        try:
            with open(
                "src/ui/components/landing_card.html", "r", encoding="utf-8"
            ) as f:
                landing_html = f.read()
            st.markdown(landing_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(str(UIComponentLoadError(e)))

    with col2:
        st.markdown('<div style="padding-top: 1rem;"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("Get Started")
            st.caption(
                "Password must be at least 6 characters. Please verify your registration in your inbox before using the app."
            )
            auth_mode = st.radio(
                "Mode",
                ["Login", "Sign Up"],
                horizontal=True,
                label_visibility="collapsed",
            )
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if auth_mode == "Login":
                if st.button("Login", use_container_width=True, type="primary"):
                    success, message = db.sign_in(email, password)
                    if success:
                        st.session_state.user_id = db.user_id
                        st.session_state.user_email = db.user_email
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            else:
                if st.button("Sign Up", use_container_width=True, type="primary"):
                    success, message = db.sign_up(email, password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

    st.stop()

# Sidebar: Account
with st.sidebar:
    st.header("Account")
    st.write(f"Logged in as: **{st.session_state.user_email}**")
    if st.button("Logout", type="primary", use_container_width=True):
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

st.markdown(
    '<h1 class="landing-title" style="font-size: 3rem; margin-top: 1rem; margin-bottom: 2rem;">Query<span class="accent-text">Graph</span></h1>',
    unsafe_allow_html=True,
)

# Ensure the database instance knows who the current user is
db.user_id = st.session_state.user_id
db.user_email = st.session_state.user_email

# Auto-load last session if starting fresh
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

# Sidebar: Controls

with st.sidebar:
    st.header("History")
    threads = db.get_threads()
    if threads:
        for t in threads:
            # Display dynamic LLM generated title if cached, else fallback to database title
            title = (
                st.session_state.thread_titles.get(t["thread_id"])
                or t.get("title")
                or t["thread_id"]
            )
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

    col_sync, col_schema = st.columns(2)
    with col_sync:
        if st.button(
            "🔄 Sync Schema",
            type="primary",
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
                    status.update(
                        label="Sync Complete!", state="complete", expanded=False
                    )
                    st.success("Schema metadata updated successfully!")
                    st.rerun()  # Rerun to refresh the timestamp display
                except Exception as e:
                    status.update(label="Sync Failed", state="error")
                    st.error(f"Sync failed: {e}")
    with col_schema:
        if st.button("Show Schema", type="primary", use_container_width=True):
            st.session_state.show_schema_modal = not st.session_state.get(
                "show_schema_modal", False
            )

    st.caption(f"Last Schema Sync: **{last_sync_display}**")

# Display Schema Modal
if st.session_state.get("show_schema_modal", False):
    st.markdown("---")
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.subheader("Database Schema")
    with col2:
        if st.button("Close", key="close_schema"):
            st.session_state.show_schema_modal = False
            st.rerun()

    try:
        tables = load_and_filter_schema(
            schema_path="olist/models/marts/_marts_schema.yml",
            exclude_tables=nodes.exclude_tables,
            exclude_columns=nodes.exclude_columns,
        )
        if not tables:
            st.warning(
                "Could not load schema from olist/models/marts/_marts_schema.yml"
            )
        else:
            st.markdown("```text\nDatabase Schema")
            table_names = list(tables.keys())
            for i, t_name in enumerate(table_names):
                prefix = "└──" if i == len(table_names) - 1 else "├──"
                st.markdown(f"{prefix} {t_name}")
            st.markdown("```")

            for t_name, t_data in tables.items():
                with st.expander(t_name):
                    # Print tree
                    tree_str = f"{t_name}\n"
                    cols = t_data["columns"]
                    for i, col in enumerate(cols):
                        prefix = "└──" if i == len(cols) - 1 else "├──"
                        tags_str = f" ({', '.join(col['tags'])})" if col["tags"] else ""
                        tree_str += f"{prefix} {col['name']}{tags_str}\n"
                    st.code(tree_str, language="text")

                    # Print ascii table
                    box_width = 50
                    table_str = f"┌{'─' * (box_width)}┐\n"
                    table_str += f"│ {t_name:<{box_width - 1}}│\n"
                    table_str += f"├{'─' * (box_width)}┤\n"
                    for col in cols:
                        tag_str = " ".join(col["tags"])
                        col_row = f"{col['name']:<25} {col['type']:<8} {tag_str:<4}"
                        table_str += f"│ {col_row:<{box_width - 1}}│\n"
                    table_str += f"└{'─' * (box_width)}┘\n"
                    st.code(table_str, language="text")
    except YAMLProcessingError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error rendering schema: {e}")

    st.markdown("---")

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

        if msg.get("thought_process"):
            with st.expander("Thought Process"):
                st.write(msg["thought_process"])
        if msg.get("sql_query"):
            with st.expander("Generated SQL"):
                st.code(msg["sql_query"], language="sql")

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

                    msg_dict = {
                        "role": "assistant",
                        "type": "dataframe",
                        "content": combined_content,
                        "full_data": full_results,
                    }
                    if result.get("generated_sql"):
                        msg_dict["sql_query"] = result["generated_sql"].get("sql_query")
                        msg_dict["thought_process"] = result["generated_sql"].get(
                            "thought_process"
                        )

                        if msg_dict["thought_process"]:
                            with st.expander("Thought Process"):
                                st.write(msg_dict["thought_process"])
                        if msg_dict["sql_query"]:
                            with st.expander("Generated SQL"):
                                st.code(msg_dict["sql_query"], language="sql")

                    st.session_state.messages.append(msg_dict)

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

                    msg_dict = {
                        "role": "assistant",
                        "type": "text",
                        "content": answer,
                        "full_data": full_results,
                    }
                    if result.get("generated_sql"):
                        msg_dict["sql_query"] = result["generated_sql"].get("sql_query")
                        msg_dict["thought_process"] = result["generated_sql"].get(
                            "thought_process"
                        )

                        if msg_dict["thought_process"]:
                            with st.expander("Thought Process"):
                                st.write(msg_dict["thought_process"])
                        if msg_dict["sql_query"]:
                            with st.expander("Generated SQL"):
                                st.code(msg_dict["sql_query"], language="sql")

                    st.session_state.messages.append(msg_dict)
                else:
                    st.error(
                        "An unexpected error occurred. No answer or table was generated."
                    )

                # Generate title after the first message in a new conversation
                if st.session_state.thread_id not in st.session_state.thread_titles:
                    user_msgs = [
                        m for m in st.session_state.messages if m.get("role") == "user"
                    ]
                    if user_msgs:
                        try:
                            fast_llm = config.get_fast_llm()
                            title = generate_chat_title(
                                fast_llm, user_msgs[0]["content"]
                            )
                            st.session_state.thread_titles[
                                st.session_state.thread_id
                            ] = title
                        except Exception:
                            pass

            except Exception as e:
                error_msg = f"Error invoking graph: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "type": "warning", "content": error_msg}
                )
            finally:
                # Ensure all traces are flushed to Langfuse before Streamlit exits or reruns
                config.get_langfuse().flush()
