# MCP servers: other programs' tools for the web UI's model

The server is an MCP client. An MCP server is a program on this computer that gives the model tools of another
application, for example [blender-mcp](https://github.com/ahujasid/blender-mcp) for Blender. The model in the web UI
then calls those tools the way it calls its own tools (`web_search`, `create_file`, ...).

It is not the other way round: arcinferzero itself is not an MCP server here.

## Setting one up

1. Settings > MCP servers > mcp.json. The format is the usual one (Claude Desktop, Cursor and VS Code use it too):

   ```json
   {
     "mcpServers": {
       "blender": { "command": "uvx", "args": ["blender-mcp"] }
     }
   }
   ```

   Save. The file is `%USERPROFILE%\.arcinferzero\mcp.json`, and you can also edit it by hand.

2. Turn the server's switch on. The switch counts for the scope of the chat on screen: its project, or every chat
   outside projects. Each tool costs prompt tokens, and small models choose worse among many tools, so switch on
   only what a project needs.

3. Send a message. At the first answer the server starts the program and reads its tools. The answer shows
   "starting MCP servers…" while that happens; `uvx` and `npx` may download the server the first time. The program
   then keeps running until you press Stop, change its entry in mcp.json, or stop the server.

### Blender (blender-mcp)

These steps come from blender-mcp's own instructions:

- install [uv](https://docs.astral.sh/uv/) so that `uvx` is on the PATH;
- in Blender, install the project's `addon.py` (Edit > Preferences > Add-ons > Install);
- in Blender's sidebar (N), "BlenderMCP" tab, press "Connect to MCP server";
- in arcinferzero, add the entry above.

The model can then read the scene, create and change objects, and run Python in Blender.

## Fields of an entry

| Field | Meaning |
|---|---|
| `command` | The program. A bare name is searched in PATH, also as `.exe`, `.cmd`, `.bat` (`npx` is `npx.cmd`). |
| `args` | Its arguments, a list of strings. |
| `env` | Extra environment variables, an object of strings. |
| `cwd` | Its working folder (default: the server's). |
| `timeout` | Seconds one tool call may take (default 300). |
| `disabled` | `true`: left out. |

Only stdio servers are supported (the program speaks newline-delimited JSON-RPC 2.0 on stdin / stdout, protocol
2025-06-18). HTTP servers (a URL instead of a command) are not supported yet.

## Safety

- A call asks first, under its card: Run, Always for this tool, or Don't run. "Always" lasts until the page is
  reloaded. A tool that the server marks read-only (`readOnlyHint`) runs without asking.
- These are programs on your own computer, so the MCP endpoints answer only connections from this machine
  (127.0.0.1). A server opened to the network with `--host` does not run them for another computer.
- The program runs as you, with your files. Add only servers you trust. Each program and everything it starts run in
  a job object, so Stop (or the server's end) ends them all.

## What the model gets

- The tools are named `<server>__<tool>`, for example `blender__get_scene_info`. Their descriptions begin with
  `[blender (MCP)]`.
- A line in the system prompt names the connected servers, followed by each server's own `instructions`, if it
  sends any.
- Results:
  - text is given as text;
  - images (a Blender viewport screenshot) are saved in the chat's files and shown to a model that sees images;
  - resources are given as their text or URI;
  - a result with `isError` is given as an error.

## API (this computer only)

| Endpoint | What it does |
|---|---|
| `GET /v1/mcp[?project=ID]` | The servers in mcp.json, each with its switch for that scope, whether it is running, its error and its tools. |
| `GET /v1/mcp/config`, `PUT /v1/mcp/config {"text"}` | The file's text. On save, servers whose entry changed or was removed are stopped. |
| `PUT /v1/mcp/enabled {"server", "on", "project"?}` | Turns a server's switch on or off. The switches are kept in `mcp-enabled.json`. |
| `POST /v1/mcp/start`, `POST /v1/mcp/stop {"server"}` | Starts or stops a server. |
| `POST /v1/mcp/tools {"project"?}` | Starts the switched-on servers and returns their tools in the chat API's form, plus the servers that failed. |
| `POST /v1/mcp/call {"server", "tool", "arguments", "chat_id"?}` | One `tools/call`. Returns `text`, `is_error`, and `files` (images saved in the chat). Closing the request cancels the call (`notifications/cancelled`). |

The code is in `tools/server_mcp.h`, and the web UI part is in `webui/index.html` (`mcpTools`, `runMcpTool`,
`mcpShow`).
