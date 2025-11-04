# 02 - Tool/Function Calling

## 📖 What is Tool Calling?

**Tool calling** (also known as function calling) allows LLMs to interact with external systems by calling predefined functions. The LLM decides *when* and *how* to use tools based on the user's request.

## 🎯 How It Works

```
User Query: "What's the weather in Paris and convert to Fahrenheit?"
      ↓
   [LLM Reasoning]
      ↓
   Decides to use: get_weather("Paris")
      ↓
   [Execute Tool] → Returns: {"temp": 20, "unit": "C"}
      ↓
   [LLM Reasoning]
      ↓
   Decides to use: convert_temp(20, "C", "F")
      ↓
   [Execute Tool] → Returns: 68
      ↓
   Final Response: "It's 68°F in Paris"
```

## 🔧 Function Definition Schema

LLMs need function schemas to understand available tools:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

## 💡 Key Concepts

### 1. Tool Definition
Define what the tool does and its parameters:
```python
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather data"""
    # Implementation
    return {"temp": 20, "condition": "sunny"}
```

### 2. Tool Registration
Register tools with the LLM:
```python
# OpenAI-style
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,  # ← Tool definitions
    tool_choice="auto"  # Let model decide
)
```

### 3. Tool Execution
Execute the tool when LLM requests it:
```python
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        # Execute the actual function
        result = execute_function(function_name, arguments)
```

### 4. Return Results
Feed results back to the LLM:
```python
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(result)
})

# LLM uses result to formulate final response
final_response = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)
```

## 🚀 Practical Example

See `weather_agent.py` for a complete working agent that:
- Accepts natural language queries
- Calls weather and unit conversion tools
- Handles multi-step reasoning
- Returns natural responses

## 🎓 Best Practices

### 1. Clear Descriptions
```python
# ❌ Bad
"description": "Gets weather"

# ✅ Good
"description": "Get current weather conditions including temperature, humidity, and conditions for a specific city"
```

### 2. Validation
```python
def get_weather(city: str, units: str = "celsius") -> dict:
    # Validate inputs
    if not city or len(city) < 2:
        return {"error": "Invalid city name"}

    if units not in ["celsius", "fahrenheit"]:
        return {"error": "Units must be 'celsius' or 'fahrenheit'"}

    # Execute
    return fetch_weather_data(city, units)
```

### 3. Error Handling
```python
try:
    result = execute_tool(name, args)
except Exception as e:
    result = {
        "error": str(e),
        "message": "Tool execution failed"
    }
```

## 📊 Common Patterns

### Sequential Tools
```
Query: "What's weather in NYC and should I bring umbrella?"
→ get_weather("NYC")
→ analyze_conditions(weather_data)
→ Response: "65°F and sunny, no umbrella needed"
```

### Parallel Tools
```
Query: "Compare weather in NYC and LA"
→ get_weather("NYC") + get_weather("LA") [parallel]
→ compare_results()
→ Response: "NYC is cooler at 65°F vs LA at 75°F"
```

### Conditional Tools
```
Query: "Get weather and convert if needed"
→ get_weather("Paris") → 20°C
→ IF user_prefers_fahrenheit: convert_temp()
→ Response: "68°F in Paris"
```

## 🔗 Resources

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/claude/docs/tool-use)
- [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/)
