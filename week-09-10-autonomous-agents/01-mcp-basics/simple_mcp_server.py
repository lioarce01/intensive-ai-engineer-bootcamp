"""
Simple MCP Server Example
Demonstrates basic MCP server setup with tools and resources
"""

import asyncio
from typing import Any
from datetime import datetime

# Simulated MCP server components (conceptual - actual SDK may differ)
class MCPServer:
    """Simplified MCP Server implementation"""

    def __init__(self, name: str):
        self.name = name
        self.tools = {}
        self.resources = {}

    def tool(self, name: str, description: str):
        """Decorator to register a tool"""
        def decorator(func):
            self.tools[name] = {
                "function": func,
                "description": description,
                "name": name
            }
            return func
        return decorator

    def resource(self, uri: str, description: str):
        """Decorator to register a resource"""
        def decorator(func):
            self.resources[uri] = {
                "function": func,
                "description": description,
                "uri": uri
            }
            return func
        return decorator

    async def call_tool(self, name: str, args: dict) -> Any:
        """Execute a tool by name"""
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")
        return await self.tools[name]["function"](**args)

    async def read_resource(self, uri: str) -> Any:
        """Read a resource by URI"""
        if uri not in self.resources:
            raise ValueError(f"Resource {uri} not found")
        return await self.resources[uri]["function"]()

    def list_tools(self) -> list:
        """List all available tools"""
        return [
            {
                "name": tool["name"],
                "description": tool["description"]
            }
            for tool in self.tools.values()
        ]

    def list_resources(self) -> list:
        """List all available resources"""
        return [
            {
                "uri": resource["uri"],
                "description": resource["description"]
            }
            for resource in self.resources.values()
        ]


# Initialize the server
server = MCPServer("demo-server")


# ===== TOOLS (Functions the LLM can call) =====

@server.tool(
    name="calculate",
    description="Perform basic arithmetic operations"
)
async def calculate(operation: str, a: float, b: float) -> dict:
    """
    Calculator tool
    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number
    """
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero"
    }

    result = operations.get(operation, "Invalid operation")

    return {
        "operation": operation,
        "operands": [a, b],
        "result": result
    }


@server.tool(
    name="get_current_time",
    description="Get the current date and time"
)
async def get_current_time() -> dict:
    """Returns current timestamp"""
    now = datetime.now()
    return {
        "timestamp": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": "UTC"
    }


@server.tool(
    name="search_docs",
    description="Search documentation database"
)
async def search_docs(query: str, limit: int = 3) -> dict:
    """
    Simulated documentation search
    Args:
        query: Search query
        limit: Maximum results to return
    """
    # Simulated search results
    mock_results = [
        {"title": f"Result for '{query}' - Doc 1", "relevance": 0.95},
        {"title": f"Result for '{query}' - Doc 2", "relevance": 0.87},
        {"title": f"Result for '{query}' - Doc 3", "relevance": 0.76},
    ]

    return {
        "query": query,
        "results": mock_results[:limit],
        "total_found": len(mock_results)
    }


# ===== RESOURCES (Data the LLM can read) =====

@server.resource(
    uri="system://status",
    description="Current system status"
)
async def system_status() -> dict:
    """System status information"""
    return {
        "status": "operational",
        "uptime": "24h 35m",
        "server_name": server.name,
        "tools_available": len(server.tools),
        "resources_available": len(server.resources)
    }


@server.resource(
    uri="docs://quickstart",
    description="Quick start documentation"
)
async def quickstart_docs() -> str:
    """Quick start guide"""
    return """
# Quick Start Guide

1. List available tools: server.list_tools()
2. Call a tool: server.call_tool(name, args)
3. Read resources: server.read_resource(uri)

Example:
    result = await server.call_tool("calculate", {
        "operation": "add",
        "a": 5,
        "b": 3
    })
"""


# ===== DEMO USAGE =====

async def demo():
    """Demonstrate MCP server capabilities"""

    print("=" * 60)
    print(f"MCP Server: {server.name}")
    print("=" * 60)

    # List available tools
    print("\n📦 Available Tools:")
    for tool in server.list_tools():
        print(f"  - {tool['name']}: {tool['description']}")

    # List available resources
    print("\n📚 Available Resources:")
    for resource in server.list_resources():
        print(f"  - {resource['uri']}: {resource['description']}")

    # Demo: Call calculator tool
    print("\n" + "=" * 60)
    print("Demo 1: Calculator Tool")
    print("=" * 60)
    calc_result = await server.call_tool("calculate", {
        "operation": "multiply",
        "a": 7,
        "b": 6
    })
    print(f"Result: {calc_result}")

    # Demo: Get current time
    print("\n" + "=" * 60)
    print("Demo 2: Current Time Tool")
    print("=" * 60)
    time_result = await server.call_tool("get_current_time", {})
    print(f"Result: {time_result}")

    # Demo: Search docs
    print("\n" + "=" * 60)
    print("Demo 3: Document Search Tool")
    print("=" * 60)
    search_result = await server.call_tool("search_docs", {
        "query": "authentication",
        "limit": 2
    })
    print(f"Result: {search_result}")

    # Demo: Read system status resource
    print("\n" + "=" * 60)
    print("Demo 4: System Status Resource")
    print("=" * 60)
    status = await server.read_resource("system://status")
    print(f"Status: {status}")

    # Demo: Read quickstart docs
    print("\n" + "=" * 60)
    print("Demo 5: Quickstart Documentation Resource")
    print("=" * 60)
    docs = await server.read_resource("docs://quickstart")
    print(docs)


if __name__ == "__main__":
    print("\n🚀 Starting MCP Server Demo...\n")
    asyncio.run(demo())
    print("\n✅ Demo completed!\n")
