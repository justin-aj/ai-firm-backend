"""
Archived MCP runner (`run_mcp_server.py`).

This is a historical copy of the MCP stdio runner. Kept for reference.

The runner below is a snapshot of how the project used to start the MCP
server over stdio so tools like the Model Context Protocol inspector and
clients could connect. It is intentionally non-operational in this
repository (kept for history/reference only).
"""

import sys

# Historical imports used by the original runner (kept for reference).
# The real runtime relied on `mcp_server.app` and the stdio transport
# provided by the `modelcontextprotocol` package.
try:
    from mcp_server import app  # type: ignore
    from modelcontextprotocol.transports.stdio import StdioTransport  # type: ignore
except Exception:
    # In the archived copy we do not require these imports to succeed.
    app = None  # type: ignore
    StdioTransport = None  # type: ignore


def run_stdio_server() -> None:
    """Start the MCP server using stdio transport (archive reference).

    Note: This function is a historical reference and is not intended to
    be executed in the active codebase. The active runner has been
    removed and replaced with a deprecation shim in the root of the
    repository.
    """
    if app is None or StdioTransport is None:
        print("Archived runner: original start logic preserved for reference.")
        return

    transport = StdioTransport(app)  # type: ignore
    try:
        # In the original code this call blocked and served MCP messages
        # over stdio until the process exited.
        transport.run()
    except Exception as exc:
        print(f"MCP runner error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    run_stdio_server()
"""
Archived MCP runner (`run_mcp_server.py`).

This is a historical copy of the MCP stdio runner. Kept for reference.
"""

# Archived contents removed for brevity.
