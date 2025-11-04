# 01 - Model Context Protocol (MCP) Basics

## 📖 What is MCP?

**Model Context Protocol (MCP)** is an open protocol that standardizes how applications provide context to Large Language Models (LLMs). Think of it as a universal adapter that lets AI models interact with different tools and data sources.

### Key Concepts

1. **MCP Server**: Exposes resources, tools, and prompts
2. **MCP Client**: Connects to servers and uses their capabilities
3. **Resources**: Data that the LLM can read (files, API responses)
4. **Tools**: Functions the LLM can call
5. **Prompts**: Pre-built prompt templates

## 🎯 Why MCP?

```
Traditional Approach:
[App] ← Custom Integration → [LLM] ← Custom Integration → [Tool]
    ↓                                                           ↓
  Brittle                                               Not Reusable

MCP Approach:
[App] ←→ [MCP Client] ←→ [MCP Server] ←→ [Tool]
         Standard        Standard         Reusable
```

**Benefits**:
- Standardized interface
- Reusable integrations
- Better security boundaries
- Easy to extend

## 🔧 Core Components

### MCP Server Structure
```python
from mcp.server import Server, Tool, Resource

# Define a tool (function the LLM can call)
@server.tool()
def get_weather(city: str) -> dict:
    return {"city": city, "temp": 72, "condition": "sunny"}

# Define a resource (data the LLM can read)
@server.resource("weather://current")
def current_weather() -> str:
    return "Current weather data..."
```

### MCP Client Usage
```python
from mcp.client import ClientSession

async with ClientSession(server_url) as session:
    # List available tools
    tools = await session.list_tools()

    # Call a tool
    result = await session.call_tool("get_weather", {"city": "NYC"})

    # Read a resource
    data = await session.read_resource("weather://current")
```

## 🚀 Quick Example

See `simple_mcp_server.py` for a complete working example.

## 📚 Key Takeaways

- MCP separates concerns: data access, tool execution, and LLM reasoning
- Servers can be written in any language that supports JSON-RPC
- One MCP server can serve multiple clients
- Security: MCP servers run in isolated contexts

## 🔗 Resources

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Example Servers](https://github.com/modelcontextprotocol/servers)
