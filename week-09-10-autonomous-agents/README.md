# Week 9-10: Autonomous Agents - MCP, Tool Calling, and Memory Systems

This folder contains materials for bootcamp weeks 9-10, focusing on building autonomous AI agents that can use tools and maintain context.

## 📋 Week Overview

**Goal**: Build an autonomous agent using Model Context Protocol (MCP) with real API integrations

**Key Topics**:
- Model Context Protocol (MCP) fundamentals
- Tool/Function calling with LLMs
- Agent orchestration patterns
- Memory systems for context management
- Multi-step reasoning
- Error handling and retries

## 📁 Project Structure

```
week-09-10-autonomous-agents/
├── README.md                        # This file
├── 01-mcp-basics/                   # MCP protocol fundamentals
│   ├── README.md                    # MCP overview
│   ├── mcp_tutorial.ipynb          # Interactive notebook
│   ├── simple_mcp_server.py        # Basic MCP server
│   └── requirements.txt            # Dependencies
├── 02-tool-calling/                 # Tool/Function calling
│   ├── README.md                    # Tool calling guide
│   ├── tool_calling_tutorial.ipynb # Interactive examples
│   ├── weather_agent.py            # Complete example
│   └── requirements.txt            # Dependencies
└── 03-agent-memory/                 # Memory systems
    ├── README.md                    # Memory patterns
    ├── memory_tutorial.ipynb       # Interactive notebook
    ├── conversation_agent.py       # Agent with memory
    └── requirements.txt            # Dependencies
```

## 🚀 Quick Start

### 1. MCP Basics
```bash
cd week-09-10-autonomous-agents/01-mcp-basics
pip install -r requirements.txt
python simple_mcp_server.py
```

### 2. Tool Calling Agent
```bash
cd week-09-10-autonomous-agents/02-tool-calling
pip install -r requirements.txt
python weather_agent.py
```

### 3. Memory Systems
```bash
cd week-09-10-autonomous-agents/03-agent-memory
pip install -r requirements.txt
python conversation_agent.py
```

## 🎯 Learning Objectives

- Understand the Model Context Protocol (MCP) specification
- Implement tool/function calling with OpenAI and Anthropic APIs
- Build agents that can use multiple tools in sequence
- Design effective memory systems for context retention
- Handle errors and edge cases in autonomous agents
- Create production-ready agent workflows

## 🛠 Technology Stack

- **Agent Frameworks**: LangChain, AutoGen
- **APIs**: OpenAI, Anthropic Claude
- **MCP**: Model Context Protocol SDK
- **Tools**: Python requests, custom functions
- **Memory**: In-memory, vector stores (FAISS)

## 📚 Resources

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Microsoft AutoGen](https://microsoft.github.io/autogen/)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

---

## 📊 Progress Tracker

- **01-MCP Basics**: ✅ Complete - Protocol fundamentals and server setup
- **02-Tool Calling**: ✅ Complete - Function calling with real APIs
- **03-Agent Memory**: ✅ Complete - Context management systems
- **04-Final Project**: 🎯 Build complete autonomous agent with MCP + tools
