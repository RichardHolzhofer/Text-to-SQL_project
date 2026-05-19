import os
import sys
import uuid
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from langgraph_sdk import get_sync_client

from mcp.server.fastmcp import FastMCP

# Ensure the root of the project is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Load environment variables
load_dotenv()

# Create FastMCP server
mcp = FastMCP("Text-to-SQL MCP")

LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://langgraph-api:8000")
USER_ID = os.getenv("MCP_USER_ID")
USER_EMAIL = os.getenv("MCP_USER_EMAIL")


@mcp.tool()
def query_olist_database(query: str, thread_id: Optional[str] = None) -> str:
    """
    Queries the Olist e-commerce database using natural language.

    Args:
        query: The natural language question to ask about the e-commerce database (e.g. sales, orders, products).
        thread_id: Optional UUID to resume a previous conversation thread. If not provided, a new thread is created.
    """
    global USER_ID, USER_EMAIL

    # Reload environment to catch fresh auth tokens without restarting container if mounted/updated
    load_dotenv(override=True)
    USER_ID = os.getenv("MCP_USER_ID")
    USER_EMAIL = os.getenv("MCP_USER_EMAIL")

    if not USER_ID or not USER_EMAIL:
        return (
            "Error: MCP_USER_ID or MCP_USER_EMAIL is not configured in the server environment.\n"
            "Please run 'python src/mcp/setup_mcp_auth.py' first to authenticate."
        )

    try:
        client = get_sync_client(url=LANGGRAPH_URL)

        # Determine the thread ID
        active_thread_id = (
            thread_id.strip()
            if (thread_id and thread_id.strip())
            else str(uuid.uuid4())
        )

        # Ensure thread exists in LangGraph
        client.threads.create(
            thread_id=active_thread_id,
            if_exists="do_nothing",
        )

        # Formulate metadata
        metadata_dict = {
            "fast_llm": os.getenv("FAST_MODEL", "gpt-4o-mini"),
            "smart_llm": os.getenv("SMART_MODEL", "gpt-4o"),
            "langfuse_session_id": active_thread_id,
            "langfuse_user_id": USER_EMAIL,
        }

        graph_config = {
            "configurable": {"thread_id": active_thread_id},
            "metadata": metadata_dict,
            "run_name": "text-to-sql-mcp",
        }

        # Invoke the graph run via LangGraph API client
        result = client.runs.wait(
            active_thread_id,
            "text_to_sql",
            input={
                "question": query,
                "user_id": USER_ID,
                "user_email": USER_EMAIL,
                "chat_history": [{"role": "user", "content": query}],
            },
            config=graph_config,
            metadata=metadata_dict,
        )

        # 1. Flagged for safety reasons
        if result.get("answer") and "flagged for safety reasons" in result["answer"]:
            return f"[Thread ID: {active_thread_id}]\n\n⚠️ Safety Warning: {result['answer']}"

        # 2. Unsupported explanation
        elif result.get("generated_sql") and result["generated_sql"].get(
            "unsupported_explanation"
        ):
            return f"[Thread ID: {active_thread_id}]\n\n❌ Query Unsupported: {result['generated_sql']['unsupported_explanation']}"

        # Generate CSV if query_results exceeds thresholds
        full_results = result.get("query_results")
        download_link = ""
        is_tabular = result.get("router") and result["router"].get("route") == "tab"
        threshold = 10 if is_tabular else 5

        if full_results and len(full_results) > threshold:
            PROJECT_ROOT = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../..")
            )
            downloads_dir = os.path.join(PROJECT_ROOT, "downloads")
            os.makedirs(downloads_dir, exist_ok=True)

            # Convert full_results to CSV
            import pandas as pd

            df_full = pd.DataFrame(full_results)
            csv_data = df_full.to_csv(index=False)

            # Formulate filename
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            email_prefix = USER_EMAIL.split("@")[0] if USER_EMAIL else "user"
            filename = f"{email_prefix}_{timestamp}.csv"
            filepath = os.path.join(downloads_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(csv_data)

            # Construct clickable file URL using host path if provided, fallback to relative path
            host_project_path = os.getenv("HOST_PROJECT_PATH")
            if host_project_path:
                # Ensure Windows path uses forward slashes for the file:// protocol
                clean_host_path = host_project_path.replace(os.sep, "/")
                if not clean_host_path.startswith("/"):
                    clean_host_path = "/" + clean_host_path
                file_uri = f"file://{clean_host_path}/downloads/{filename}"
                download_link = f"\n\n📥 **The full dataset contains {len(full_results)} records.** It has been exported to your workspace: [Open Full Results CSV]({file_uri}) (saved in your `downloads/` folder as `{filename}`)."
            else:
                download_link = f"\n\n📥 **The full dataset contains {len(full_results)} records.** It has been exported to your project workspace inside the container `downloads/{filename}`."

        # 3. Tabular answer
        if is_tabular and result.get("tabular_answer") is not None:
            tab_resp = result["tabular_answer"]
            answer = tab_resp.get("answer", "")
            data = tab_resp.get("data", [])

            # Format table nicely in markdown
            import pandas as pd

            df = pd.DataFrame(data)
            markdown_table = df.to_markdown(index=False)

            response = f"[Thread ID: {active_thread_id}]\n\n"
            if result.get("generated_sql") and result["generated_sql"].get(
                "fuzzy_match_warning"
            ):
                response += (
                    f"⚠️ Warning: {result['generated_sql']['fuzzy_match_warning']}\n\n"
                )

            if answer:
                response += f"{answer}\n\n"
            response += f"{markdown_table}"
            response += download_link
            return response

        # 4. Fallback to Natural Language Answer
        elif result.get("answer"):
            response = f"[Thread ID: {active_thread_id}]\n\n"
            if result.get("generated_sql") and result["generated_sql"].get(
                "fuzzy_match_warning"
            ):
                response += (
                    f"⚠️ Warning: {result['generated_sql']['fuzzy_match_warning']}\n\n"
                )
            response += result["answer"]
            response += download_link
            return response

        else:
            return f"[Thread ID: {active_thread_id}]\n\nError: An unexpected response structure was returned by the backend graph."

    except Exception as e:
        return f"Error invoking text-to-sql backend graph: {str(e)}"


if __name__ == "__main__":
    print("Starting Text-to-SQL FastMCP Server...", file=sys.stderr)
    import uvicorn

    # FastMCP exposes an ASGI application via mcp.sse_app()
    # We use a try-except fallback to support different versions of the MCP python SDK
    try:
        app = mcp.sse_app()
    except AttributeError:
        try:
            app = mcp.http_app()
        except AttributeError:
            try:
                app = mcp.asgi()
            except AttributeError:
                try:
                    app = mcp.create_asgi_app()
                except AttributeError:
                    app = mcp

    uvicorn.run(app, host="0.0.0.0", port=8000)
