"""
Conversational Agent with Memory
Demonstrates different memory patterns for maintaining context
"""

import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict


# ===== MEMORY IMPLEMENTATIONS =====

class BufferMemory:
    """
    Simple buffer that stores recent messages
    Pros: Fast, simple
    Cons: Limited by max messages
    """

    def __init__(self, max_messages: int = 10):
        self.messages: List[Dict[str, str]] = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str):
        """Add a message to memory"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all stored messages"""
        return self.messages

    def clear(self):
        """Clear all messages"""
        self.messages = []


class EntityMemory:
    """
    Extracts and stores entities (facts) from conversation
    Pros: Structured, queryable
    Cons: Requires extraction logic
    """

    def __init__(self):
        self.entities: Dict[str, Any] = {
            "user_name": None,
            "user_preferences": {},
            "mentioned_topics": [],
            "mentioned_locations": [],
            "facts": []
        }

    def extract_and_store(self, role: str, content: str):
        """Extract entities from message and store"""
        content_lower = content.lower()

        # Extract name
        name_patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"call me (\w+)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, content_lower)
            if match:
                self.entities["user_name"] = match.group(1).capitalize()

        # Extract preferences
        if "prefer" in content_lower or "like" in content_lower:
            self.entities["facts"].append({
                "type": "preference",
                "content": content,
                "timestamp": datetime.now().isoformat()
            })

        # Extract locations (simplified)
        cities = ["paris", "london", "new york", "tokyo", "berlin", "sydney"]
        for city in cities:
            if city in content_lower and city not in self.entities["mentioned_locations"]:
                self.entities["mentioned_locations"].append(city.title())

        # Extract topics
        topics = ["weather", "food", "travel", "sports", "music", "movies"]
        for topic in topics:
            if topic in content_lower and topic not in self.entities["mentioned_topics"]:
                self.entities["mentioned_topics"].append(topic)

    def get_entities(self) -> Dict[str, Any]:
        """Get all stored entities"""
        return self.entities

    def get_user_context(self) -> str:
        """Generate user context summary"""
        ctx = []
        if self.entities["user_name"]:
            ctx.append(f"User name: {self.entities['user_name']}")
        if self.entities["mentioned_topics"]:
            ctx.append(f"Interested in: {', '.join(self.entities['mentioned_topics'])}")
        if self.entities["mentioned_locations"]:
            ctx.append(f"Mentioned places: {', '.join(self.entities['mentioned_locations'])}")

        return " | ".join(ctx) if ctx else "No user context yet"


class HybridMemory:
    """
    Combines buffer and entity memory for robust context management
    """

    def __init__(self, max_buffer: int = 10):
        self.buffer = BufferMemory(max_messages=max_buffer)
        self.entities = EntityMemory()
        self.conversation_summary = ""

    def add_interaction(self, user_msg: str, agent_msg: str):
        """Add a complete interaction (user + agent)"""
        # Add to buffer
        self.buffer.add_message("user", user_msg)
        self.buffer.add_message("assistant", agent_msg)

        # Extract entities
        self.entities.extract_and_store("user", user_msg)
        self.entities.extract_and_store("assistant", agent_msg)

    def get_full_context(self) -> Dict[str, Any]:
        """Get complete context for agent"""
        return {
            "recent_messages": self.buffer.get_messages()[-5:],
            "entities": self.entities.get_entities(),
            "user_summary": self.entities.get_user_context()
        }

    def format_for_prompt(self) -> str:
        """Format memory as text for LLM prompt"""
        context = self.get_full_context()

        prompt = f"""# Conversation Context

## User Profile
{context['user_summary']}

## Recent Messages
"""
        for msg in context['recent_messages']:
            prompt += f"{msg['role'].upper()}: {msg['content']}\n"

        return prompt


# ===== CONVERSATIONAL AGENT =====

class ConversationalAgent:
    """
    Agent with memory that can maintain context across interactions
    """

    def __init__(self, memory_type: str = "hybrid"):
        """
        Initialize agent with specified memory type
        Args:
            memory_type: 'buffer', 'entity', or 'hybrid'
        """
        if memory_type == "buffer":
            self.memory = BufferMemory()
        elif memory_type == "entity":
            self.memory = EntityMemory()
        else:
            self.memory = HybridMemory()

        self.memory_type = memory_type

    def process_message(self, user_input: str) -> str:
        """
        Process user input and generate response using memory
        Args:
            user_input: User's message
        Returns:
            Agent's response
        """
        # Generate response based on memory type
        if self.memory_type == "hybrid":
            response = self._generate_context_aware_response(user_input)
            self.memory.add_interaction(user_input, response)
        elif self.memory_type == "buffer":
            self.memory.add_message("user", user_input)
            response = self._generate_simple_response(user_input)
            self.memory.add_message("assistant", response)
        else:  # entity
            self.memory.extract_and_store("user", user_input)
            response = self._generate_entity_based_response(user_input)

        return response

    def _generate_context_aware_response(self, user_input: str) -> str:
        """Generate response using full context (hybrid memory)"""
        context = self.memory.get_full_context()
        entities = context["entities"]

        user_input_lower = user_input.lower()

        # Personalized responses based on memory
        if "my name" in user_input_lower:
            if entities["user_name"]:
                return f"I remember! Your name is {entities['user_name']}."
            return "Nice to meet you! I'll remember that."

        if "what" in user_input_lower and "name" in user_input_lower:
            if entities["user_name"]:
                return f"Your name is {entities['user_name']}."
            return "I don't think you've told me your name yet."

        if "what" in user_input_lower and "talked about" in user_input_lower:
            if entities["mentioned_topics"]:
                return f"We've discussed: {', '.join(entities['mentioned_topics'])}."
            return "This is the beginning of our conversation!"

        if "where" in user_input_lower:
            if entities["mentioned_locations"]:
                return f"You've mentioned these places: {', '.join(entities['mentioned_locations'])}."
            return "We haven't talked about any specific locations yet."

        # Check recent context for follow-up questions
        recent = context["recent_messages"]
        if recent and len(recent) >= 2:
            last_user_msg = recent[-2]["content"] if len(recent) >= 2 else ""

            # Handle pronoun references
            if user_input_lower in ["what about it?", "tell me more", "continue"]:
                if "weather" in last_user_msg.lower():
                    return "The weather has been quite nice lately! Would you like forecasts for a specific city?"

        # Default response with context awareness
        greeting_words = ["hello", "hi", "hey", "greetings"]
        if any(word in user_input_lower for word in greeting_words):
            if entities["user_name"]:
                return f"Hello again, {entities['user_name']}! How can I help you today?"
            return "Hello! How can I help you?"

        return "I understand. Could you tell me more?"

    def _generate_simple_response(self, user_input: str) -> str:
        """Generate basic response (buffer memory)"""
        messages = self.memory.get_messages()
        context_size = len(messages)

        if "name" in user_input.lower():
            return f"Thanks for sharing! (I have {context_size} messages in memory)"

        return f"I see. (Conversation has {context_size} messages)"

    def _generate_entity_based_response(self, user_input: str) -> str:
        """Generate response using entities only"""
        entities = self.memory.get_entities()

        if entities["user_name"]:
            return f"Hello {entities['user_name']}! I know you're interested in: {', '.join(entities['mentioned_topics']) or 'nothing specific yet'}"

        return "Hello! Tell me about yourself."

    def show_memory_state(self):
        """Display current memory state"""
        print("\n" + "=" * 60)
        print("💾 MEMORY STATE")
        print("=" * 60)

        if self.memory_type == "hybrid":
            context = self.memory.get_full_context()
            print(f"\n📝 User Profile: {context['user_summary']}")
            print(f"\n🗂️  Entities: {json.dumps(context['entities'], indent=2)}")
            print(f"\n💬 Recent Messages ({len(context['recent_messages'])}):")
            for msg in context['recent_messages']:
                print(f"   {msg['role']}: {msg['content'][:50]}...")
        elif self.memory_type == "buffer":
            print(f"\n💬 Messages in buffer: {len(self.memory.get_messages())}")
            for msg in self.memory.get_messages():
                print(f"   {msg['role']}: {msg['content'][:50]}...")
        else:
            print(f"\n🗂️  Entities: {json.dumps(self.memory.get_entities(), indent=2)}")

        print("=" * 60)


# ===== DEMO =====

def demo():
    """Run interactive demo"""
    print("=" * 70)
    print("💬 Conversational Agent with Memory Demo")
    print("=" * 70)
    print("\nThis agent remembers:")
    print("  • Your name")
    print("  • Topics discussed")
    print("  • Locations mentioned")
    print("  • Conversation flow")
    print("\nType 'memory' to see memory state, 'quit' to exit")
    print("=" * 70)

    # Create agent with hybrid memory
    agent = ConversationalAgent(memory_type="hybrid")

    # Demo conversation flow
    demo_messages = [
        "Hello!",
        "My name is Alice",
        "I'm interested in weather",
        "I'd like to visit Paris",
        "What's my name?",
        "What have we talked about?",
        "Where do I want to visit?",
    ]

    print("\n🤖 Running automated demo...\n")

    for user_msg in demo_messages:
        print(f"👤 User: {user_msg}")
        response = agent.process_message(user_msg)
        print(f"🤖 Agent: {response}\n")

    # Show final memory state
    agent.show_memory_state()

    print("\n" + "=" * 70)
    print("✨ Demo completed!")
    print("\nTry running the interactive mode by uncommenting interactive_demo()")
    print("=" * 70)


def interactive_demo():
    """Run interactive conversation"""
    agent = ConversationalAgent(memory_type="hybrid")

    print("\n🤖 Agent: Hello! I'm a conversational agent with memory. How can I help you?")

    while True:
        user_input = input("\n👤 You: ").strip()

        if user_input.lower() == "quit":
            print("\n🤖 Agent: Goodbye! It was nice talking to you.")
            break

        if user_input.lower() == "memory":
            agent.show_memory_state()
            continue

        if not user_input:
            continue

        response = agent.process_message(user_input)
        print(f"🤖 Agent: {response}")


if __name__ == "__main__":
    # Run automated demo
    demo()

    # Uncomment for interactive mode:
    # interactive_demo()
