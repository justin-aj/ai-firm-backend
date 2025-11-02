"""
Test MCP Server Tool Registration
Verifies all 11 tools are properly registered
"""

import asyncio
from mcp_server import app


async def test_tools():
    """Test that all MCP tools are registered"""
    # Access the registered handler directly
    from mcp_server import list_tools
    
    tools = await list_tools()
    
    print(f"\n✅ Total MCP Tools Registered: {len(tools)}\n")
    
    expected_tools = [
        "sequential_thinking",
        "google_search",
        "google_search_urls_only",
        "google_image_search",
        "lm_studio_chat",
        "lm_studio_completion",
        "scrape_url",
        "scrape_urls_batch",
        "generate_embedding",
        "scrape_and_embed",
        "semantic_search"
    ]
    
    print("Registered Tools:")
    print("-" * 60)
    for i, tool in enumerate(tools, 1):
        status = "✅" if tool.name in expected_tools else "❌"
        print(f"{status} {i}. {tool.name}")
        print(f"   {tool.description[:80]}...")
        print()
    
    # Check for missing tools
    registered_names = {tool.name for tool in tools}
    missing = set(expected_tools) - registered_names
    
    if missing:
        print("\n❌ Missing Tools:")
        for tool in missing:
            print(f"   - {tool}")
    else:
        print("\n✅ All expected tools are registered!")
    
    # Check for unexpected tools
    unexpected = registered_names - set(expected_tools)
    if unexpected:
        print("\n⚠️  Unexpected Tools:")
        for tool in unexpected:
            print(f"   - {tool}")
    
    print("\n" + "=" * 60)
    print(f"Summary: {len(tools)} tools registered")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_tools())
