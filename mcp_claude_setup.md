# 🔨 Text-to-SQL MCP Setup Guide (Claude Desktop)

This guide walks you through setting up and connecting the custom **Text-to-SQL Model Context Protocol (MCP)** server to **Claude Desktop** on Windows. 

Once configured, Claude Desktop will be able to query the Olist e-commerce database using natural language, format results in markdown, and export massive datasets directly into your workspace.

---

## 📋 Prerequisites

Before starting, ensure you have the following installed on your host system:
1. **Docker Desktop** (with WSL2 backend enabled).
2. **Node.js** (includes `npx` which is used for the MCP-remote bridge).
3. **Claude Desktop** (installed and logged in).

---

## 🔑 Step 1: Authenticate and Set Up Your Session

The database requires user authentication to secure calls and trace execution (via Langfuse). Run the interactive local CLI setup script to authenticate and write your credentials to `.env`:

```bash
# Run the local python setup script using uv
uv run python src/mcp/setup_mcp_auth.py
```

This script will:
1. Securely log you into Supabase using your developer email and password.
2. Automatically write `MCP_USER_ID` and `MCP_USER_EMAIL` into your `.env` file.
3. List your **15 most recent chat threads** (with their IDs) so you can easily copy and paste one to continue a previous conversation in the MCP server!

---

## 🚀 Step 2: Start the Docker Services

Now that your credentials are saved in your `.env` file, spin up the background database services, Redis backend, LangGraph API executor, and the FastMCP server. The container will automatically load your credentials from the start!

```bash
# Navigate to the project root and spin up the MCP stack
docker-compose -f docker-compose.mcp.yml up --build -d
```

Verify that all three services are running and healthy:
* `text-to-sql-redis` (Redis container)
* `text-to-sql-api` (LangGraph API executor)
* `text-to-sql-mcp` (FastMCP ASGI server exposed on port `8080`)

---

## ⚙️ Step 3: Configure Claude Desktop

To connect Claude Desktop to your local MCP server, we will add its configuration to Claude's JSON settings. You can open this file directly from the Claude Desktop UI:

1. Open **Claude Desktop**.
2. Click on your profile picture/name in the bottom-left corner and select **Settings** (or click the gear icon).
3. Navigate to the **Developer** (or **MCP**) tab.
4. Click the **"Open Config File"** (or **"Edit Config"**) button. This will automatically open your local `claude_desktop_config.json` file in your default system text editor (e.g., Notepad, VS Code).
5. Add the `text-to-sql` server configuration under the `"mcpServers"` block:

```json
{
  "mcpServers": {
    "text-to-sql": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "http://127.0.0.1:8080/sse",
        "--allow-http"
      ]
    }
  }
}
```

6. Save the file.

---

## ⚡ Step 4: Restart Claude Desktop and Start Querying!

1. Completely close **Claude Desktop** (make sure to quit from the Windows system tray in the bottom right corner!).
2. Launch Claude Desktop.
3. **Verify the Connection:** Start a new chat, click the **`+` (plus) button** in the chat input bar, and hover over **"Connectors"**. You should see **`text-to-sql`** listed there as an active, connected connector!
4. Ask Claude any database question! The tools will be called automatically in the background when needed.

### Example Prompts to Try:
* *"Show me a list of the top 5 product categories by revenue."*
* *"How many orders were delivered in state SP in 2018?"*

---

## 📥 Capped Boundaries and CSV Exports

To prevent overwhelming the chat screen with thousands of rows of data, the MCP server automatically enforces thresholds:
* **Tabular Answers:** If the query returns `> 10` rows.
* **Natural Language Answers:** If the query returns `> 5` rows.

If a query exceeds these thresholds:
1. The container generates a complete CSV file of the database results.
2. Because your local `./downloads` folder is mounted to the container, the CSV file is instantly written directly to your host machine at `<your-project-root>/downloads/`.
3. Claude will print a clickable, local file URL based on your configured `HOST_PROJECT_PATH` in `.env`:
   `[Open Full Results CSV](file:///<your-absolute-project-path>/downloads/user_timestamp.csv)`
4. Simply **click the link** in the Claude Desktop chat window, and your local operating system will instantly open the complete CSV in your default viewer (e.g., Excel, VS Code, or Cursor)!

---

## 🛑 Stopping the Services (Resource Cleanup)

Since the containers are configured to run persistently, they will automatically spin up and consume system resources (CPU, RAM, and database connection pools) whenever Docker Desktop starts (e.g., on system boot).

When you are finished querying and want to release your local system resources, shut down the background services:

```bash
# Tear down the stack and stop all services
docker-compose -f docker-compose.mcp.yml down
```

