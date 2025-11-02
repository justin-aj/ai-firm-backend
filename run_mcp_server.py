#!/usr/bin/env python3
"""
Standalone MCP Server Runner

Run this to start the MCP server for use with Claude Desktop or other MCP clients.
Usage: python run_mcp_server.py
"""

import asyncio
from mcp.server.stdio import stdio_server
from mcp_server import app


async def main():
    """Run the MCP server using stdio transport"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
