"""
Weather Agent with Tool Calling
Demonstrates autonomous agent using multiple tools to answer queries
"""

import json
from typing import Any, Dict, List
from datetime import datetime


# ===== AVAILABLE TOOLS =====

def get_weather(city: str, units: str = "celsius") -> Dict[str, Any]:
    """
    Get current weather for a city
    Args:
        city: City name
        units: Temperature units ('celsius' or 'fahrenheit')
    """
    # Simulated weather data
    weather_db = {
        "paris": {"temp": 20, "condition": "sunny", "humidity": 65},
        "london": {"temp": 15, "condition": "rainy", "humidity": 80},
        "new york": {"temp": 25, "condition": "cloudy", "humidity": 70},
        "tokyo": {"temp": 28, "condition": "sunny", "humidity": 75},
        "sydney": {"temp": 22, "condition": "partly cloudy", "humidity": 60},
    }

    city_lower = city.lower()

    if city_lower not in weather_db:
        return {
            "error": f"Weather data not available for {city}",
            "available_cities": list(weather_db.keys())
        }

    data = weather_db[city_lower]

    # Convert temperature if needed
    temp = data["temp"]
    if units == "fahrenheit":
        temp = (temp * 9/5) + 32

    return {
        "city": city,
        "temperature": round(temp, 1),
        "units": units,
        "condition": data["condition"],
        "humidity": data["humidity"],
        "timestamp": datetime.now().isoformat()
    }


def convert_temperature(temp: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    """
    Convert temperature between Celsius and Fahrenheit
    Args:
        temp: Temperature value
        from_unit: Source unit ('celsius' or 'fahrenheit')
        to_unit: Target unit ('celsius' or 'fahrenheit')
    """
    if from_unit == to_unit:
        return {"temperature": temp, "unit": to_unit}

    if from_unit == "celsius" and to_unit == "fahrenheit":
        converted = (temp * 9/5) + 32
    elif from_unit == "fahrenheit" and to_unit == "celsius":
        converted = (temp - 32) * 5/9
    else:
        return {"error": "Invalid units. Use 'celsius' or 'fahrenheit'"}

    return {
        "original": {"value": temp, "unit": from_unit},
        "converted": {"value": round(converted, 1), "unit": to_unit}
    }


def get_clothing_recommendation(temp: float, condition: str, units: str = "celsius") -> Dict[str, Any]:
    """
    Recommend clothing based on weather
    Args:
        temp: Temperature
        condition: Weather condition
        units: Temperature units
    """
    # Convert to Celsius for comparison
    temp_c = temp if units == "celsius" else (temp - 32) * 5/9

    # Determine clothing
    if temp_c < 0:
        outfit = "Heavy winter coat, gloves, and warm boots"
    elif temp_c < 10:
        outfit = "Warm jacket and layers"
    elif temp_c < 20:
        outfit = "Light jacket or sweater"
    elif temp_c < 30:
        outfit = "T-shirt and comfortable pants"
    else:
        outfit = "Light, breathable clothing"

    # Add weather-specific items
    extras = []
    if "rain" in condition.lower():
        extras.append("umbrella")
    if "sunny" in condition.lower() and temp_c > 20:
        extras.append("sunglasses")
        extras.append("sunscreen")

    return {
        "temperature": temp,
        "units": units,
        "condition": condition,
        "recommendation": outfit,
        "extras": extras
    }


# ===== TOOL DEFINITIONS (LLM Schema) =====

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions for a specific city including temperature, condition, and humidity",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name (e.g., 'Paris', 'New York')"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units (default: celsius)"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_temperature",
            "description": "Convert temperature between Celsius and Fahrenheit",
            "parameters": {
                "type": "object",
                "properties": {
                    "temp": {
                        "type": "number",
                        "description": "Temperature value to convert"
                    },
                    "from_unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Source temperature unit"
                    },
                    "to_unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Target temperature unit"
                    }
                },
                "required": ["temp", "from_unit", "to_unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_clothing_recommendation",
            "description": "Get clothing recommendations based on weather conditions",
            "parameters": {
                "type": "object",
                "properties": {
                    "temp": {
                        "type": "number",
                        "description": "Current temperature"
                    },
                    "condition": {
                        "type": "string",
                        "description": "Weather condition (e.g., 'sunny', 'rainy')"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units"
                    }
                },
                "required": ["temp", "condition"]
            }
        }
    }
]


# ===== TOOL EXECUTION ENGINE =====

# Map function names to actual Python functions
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "convert_temperature": convert_temperature,
    "get_clothing_recommendation": get_clothing_recommendation
}


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool by name with given arguments
    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of arguments
    """
    if tool_name not in TOOL_FUNCTIONS:
        return {"error": f"Tool '{tool_name}' not found"}

    try:
        func = TOOL_FUNCTIONS[tool_name]
        result = func(**arguments)
        return result
    except Exception as e:
        return {
            "error": f"Tool execution failed: {str(e)}",
            "tool": tool_name,
            "arguments": arguments
        }


# ===== SIMULATED LLM AGENT =====

class SimpleWeatherAgent:
    """
    Simulated agent that decides which tools to use
    In production, this would be replaced by actual LLM API calls
    """

    def __init__(self):
        self.conversation_history = []

    def process_query(self, query: str) -> str:
        """
        Process user query and decide which tools to call
        Args:
            query: User's natural language query
        """
        query_lower = query.lower()

        print(f"\n🤖 Agent received: '{query}'")
        print("🧠 Analyzing query and selecting tools...")

        # Simple rule-based logic (in production, LLM decides this)
        tool_calls = []

        # Pattern 1: Weather query
        if "weather" in query_lower:
            # Extract city (simplified)
            cities = ["paris", "london", "new york", "tokyo", "sydney"]
            for city in cities:
                if city in query_lower:
                    units = "fahrenheit" if "fahrenheit" in query_lower or "°f" in query_lower else "celsius"
                    tool_calls.append(("get_weather", {"city": city, "units": units}))
                    break

        # Pattern 2: Clothing recommendation
        if "wear" in query_lower or "clothing" in query_lower or "outfit" in query_lower:
            # If weather was fetched, use that data
            if tool_calls and tool_calls[0][0] == "get_weather":
                # Will add clothing recommendation after weather fetch
                pass

        # Execute tools
        results = []
        for tool_name, args in tool_calls:
            print(f"🔧 Calling tool: {tool_name}({args})")
            result = execute_tool(tool_name, args)
            results.append((tool_name, result))
            print(f"✅ Result: {json.dumps(result, indent=2)}")

            # Chain: If weather was fetched and query mentions clothing
            if tool_name == "get_weather" and not result.get("error"):
                if "wear" in query_lower or "clothing" in query_lower or "outfit" in query_lower:
                    print(f"🔧 Calling tool: get_clothing_recommendation")
                    clothing_args = {
                        "temp": result["temperature"],
                        "condition": result["condition"],
                        "units": result["units"]
                    }
                    clothing_result = execute_tool("get_clothing_recommendation", clothing_args)
                    results.append(("get_clothing_recommendation", clothing_result))
                    print(f"✅ Result: {json.dumps(clothing_result, indent=2)}")

        # Generate response (in production, LLM generates this)
        return self._generate_response(query, results)

    def _generate_response(self, query: str, results: List[tuple]) -> str:
        """Generate natural language response from tool results"""
        if not results:
            return "I couldn't find the information you're looking for."

        response_parts = []

        for tool_name, result in results:
            if result.get("error"):
                response_parts.append(f"Error: {result['error']}")
            elif tool_name == "get_weather":
                response_parts.append(
                    f"In {result['city']}, it's currently {result['temperature']}°{result['units'][0].upper()} "
                    f"and {result['condition']} with {result['humidity']}% humidity."
                )
            elif tool_name == "get_clothing_recommendation":
                response_parts.append(
                    f"I recommend wearing: {result['recommendation']}."
                )
                if result['extras']:
                    response_parts.append(f"Don't forget: {', '.join(result['extras'])}.")

        return " ".join(response_parts)


# ===== DEMO =====

def demo():
    """Run demo queries"""
    agent = SimpleWeatherAgent()

    queries = [
        "What's the weather in Paris?",
        "What's the weather in Tokyo and what should I wear?",
        "Get weather for London in Fahrenheit",
        "Weather in Sydney and clothing recommendations",
    ]

    print("=" * 70)
    print("🌤️  Weather Agent with Tool Calling Demo")
    print("=" * 70)

    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Query {i}/{len(queries)}")
        print("=" * 70)

        response = agent.process_query(query)

        print(f"\n💬 Agent response:")
        print(f"   {response}")

    print(f"\n{'=' * 70}")
    print("✅ Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
