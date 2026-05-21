import os
import sys
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
import uvicorn
from langgraph_sdk import get_sync_client
from mcp.server.fastmcp import FastMCP

from src.config.config import Config
from src.database.db import SupabaseDB
from src.exceptions.exception import (
    GraphResponseError,
    LangGraphExecutionError,
    MCPConfigError,
    PipelineException,
    SupabaseAuthError,
)


class TextToSQLMCPServer:
    def __init__(self):
        """
        Initialize the Text-to-SQL MCP Server and dynamically register its tools.
        """
        self.mcp = FastMCP("Text-to-SQL MCP")
        self.config = Config("text-to-sql-mcp")

        # Register tools dynamically within the server instance context
        @self.mcp.tool()
        def query_olist_database(query: str, thread_id: Optional[str] = None) -> str:
            """
            Queries the Olist e-commerce database using natural language.

            Args:
                query: The natural language question to ask about the e-commerce database (e.g. sales, orders, products).
                thread_id: Optional UUID to resume a previous conversation thread. If not provided, a new thread is created.
            """
            return self.query_database(query, thread_id)

    def query_database(self, query: str, thread_id: Optional[str] = None) -> str:
        """
        Business logic implementation for querying the e-commerce database.
        """
        # Reload environment dynamically to catch fresh auth tokens without restarting container
        user_id, user_email, user_password = self.config.reload_mcp_credentials()

        # If user_id is pinned in .env, we can optionally trust it, but we prioritize silent auth if email/password is present
        if not user_email or not user_password:
            # Fallback to pure pinned credentials if password isn't specified
            if user_id and user_email:
                active_user_id = user_id
                active_user_email = user_email
            else:
                raise MCPConfigError(
                    Exception(
                        "MCP_USER_EMAIL or MCP_USER_PASSWORD is not configured in the server environment. "
                        "Please configure them in your local .env configuration file."
                    )
                )
        else:
            # Run silent dynamic login/registration sequence
            try:
                db = SupabaseDB(self.config)
                success, message = db.sign_in(user_email, user_password)

                if not success:
                    print(
                        f"Login failed: {message}. Attempting automatic registration...",
                        file=sys.stderr,
                    )
                    signup_success, signup_message = db.sign_up(
                        user_email, user_password
                    )

                    if not signup_success:
                        raise SupabaseAuthError(
                            Exception(f"Dynamic registration failed: {signup_message}")
                        )

                    # Try logging in again after registration
                    success, message = db.sign_in(user_email, user_password)
                    if not success:
                        raise SupabaseAuthError(
                            Exception(
                                f"Login failed after successful registration: {message}"
                            )
                        )

                active_user_id = db.user_id
                active_user_email = db.user_email

            except Exception as auth_err:
                err_msg = str(auth_err).lower()
                if (
                    "confirm" in err_msg
                    or "verify" in err_msg
                    or "email not confirmed" in err_msg
                ):
                    raise MCPConfigError(
                        Exception(
                            f"Authentication Pending: Your new account has been registered successfully, "
                            f"but email verification is pending! "
                            f"Please check your inbox ({user_email}) for a verification link from Supabase. "
                            f"Once you click the link to confirm your email, simply ask your question again here!"
                        )
                    )
                raise MCPConfigError(
                    Exception(f"Silent user authentication failed: {str(auth_err)}")
                )

        try:
            try:
                client = get_sync_client(url=self.config.langgraph_url)

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
            except Exception as e:
                raise LangGraphExecutionError(e)

            # Formulate metadata
            metadata_dict = {
                "fast_llm": os.getenv("FAST_MODEL", "gpt-4o-mini"),
                "smart_llm": os.getenv("SMART_MODEL", "gpt-4o"),
                "langfuse_session_id": active_thread_id,
                "langfuse_user_id": active_user_email,
            }

            graph_config = {
                "configurable": {"thread_id": active_thread_id},
                "metadata": metadata_dict,
                "run_name": "text-to-sql-mcp",
            }

            # Invoke the graph run via LangGraph API client
            try:
                result = client.runs.wait(
                    active_thread_id,
                    "text_to_sql",
                    input={
                        "question": query,
                        "user_id": active_user_id,
                        "user_email": active_user_email,
                        "chat_history": [{"role": "user", "content": query}],
                    },
                    config=graph_config,
                    metadata=metadata_dict,
                )
            except Exception as e:
                raise LangGraphExecutionError(e)

            # 1. Flagged for safety reasons
            if (
                result.get("answer")
                and "flagged for safety reasons" in result["answer"]
            ):
                return f"[Thread ID: {active_thread_id}] Safety Warning: {result['answer']}"

            # 2. Unsupported explanation
            elif result.get("generated_sql") and result["generated_sql"].get(
                "unsupported_explanation"
            ):
                return f"[Thread ID: {active_thread_id}] Query Unsupported: {result['generated_sql']['unsupported_explanation']}"

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
                df_full = pd.DataFrame(full_results)
                csv_data = df_full.to_csv(index=False)

                # Formulate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                email_prefix = (
                    active_user_email.split("@")[0] if active_user_email else "user"
                )
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
                    download_link = f" The full dataset contains {len(full_results)} records. It has been exported to your workspace: [Open Full Results CSV]({file_uri}) (saved in your `downloads/` folder as `{filename}`)."
                else:
                    download_link = f" The full dataset contains {len(full_results)} records. It has been exported to your project workspace inside the container `downloads/{filename}`."

            # 3. Tabular answer
            if is_tabular and result.get("tabular_answer") is not None:
                tab_resp = result["tabular_answer"]
                answer = tab_resp.get("answer", "")
                data = tab_resp.get("data", [])

                # Format table nicely in markdown
                df = pd.DataFrame(data)
                markdown_table = df.to_markdown(index=False)

                response = f"[Thread ID: {active_thread_id}] "
                if result.get("generated_sql") and result["generated_sql"].get(
                    "fuzzy_match_warning"
                ):
                    response += (
                        f"Warning: {result['generated_sql']['fuzzy_match_warning']} "
                    )

                if answer:
                    response += f"{answer}\n\n"
                response += f"{markdown_table}"
                response += download_link
                return response

            # 4. Fallback to Natural Language Answer
            elif result.get("answer"):
                response = f"[Thread ID: {active_thread_id}] "
                if result.get("generated_sql") and result["generated_sql"].get(
                    "fuzzy_match_warning"
                ):
                    response += (
                        f"Warning: {result['generated_sql']['fuzzy_match_warning']} "
                    )
                response += result["answer"]
                if download_link:
                    response += f"\n\n{download_link}"
                return response

            else:
                raise GraphResponseError(
                    Exception(
                        "An unexpected or empty response structure was returned by the backend graph."
                    )
                )

        except PipelineException as e:
            return f"Pipeline Error: {str(e)}"
        except Exception as e:
            return f"Unexpected Error: {str(e)}"

    def get_asgi_app(self):
        """
        Creates and returns the Starlette ASGI application with cascading SDK fallbacks.
        """
        try:
            return self.mcp.sse_app()
        except AttributeError:
            try:
                return self.mcp.http_app()
            except AttributeError:
                try:
                    return self.mcp.asgi()
                except AttributeError:
                    try:
                        return self.mcp.create_asgi_app()
                    except AttributeError:
                        return self.mcp


if __name__ == "__main__":
    print("Starting Text-to-SQL FastMCP Server...", file=sys.stderr)
    server = TextToSQLMCPServer()
    app = server.get_asgi_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
