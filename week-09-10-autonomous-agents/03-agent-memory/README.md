# 03 - Agent Memory Systems

## 📖 What is Agent Memory?

**Agent memory** enables AI agents to maintain context across multiple interactions, remember past conversations, and make decisions based on historical data.

## 🎯 Why Memory Matters

```
Without Memory:
User: "My name is Alice"
Agent: "Nice to meet you, Alice!"
User: "What's my name?"
Agent: "I don't know." ❌

With Memory:
User: "My name is Alice"
Agent: "Nice to meet you, Alice!"
User: "What's my name?"
Agent: "Your name is Alice." ✅
```

## 🧠 Types of Memory

### 1. Short-Term Memory (Conversation Buffer)
Stores recent messages in the current conversation.

```python
memory = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Tell me about memory"}
]
```

**Pros**: Simple, fast
**Cons**: Limited by context window

### 2. Summary Memory
Periodically summarizes conversation to save tokens.

```python
# Original conversation (1000 tokens)
messages = [msg1, msg2, msg3, ..., msg50]

# Summarized (100 tokens)
summary = "User asked about weather in Paris. Agent provided current conditions..."
memory = [{"role": "system", "content": summary}, recent_messages]
```

**Pros**: Handles long conversations
**Cons**: May lose details

### 3. Entity Memory
Tracks specific entities (people, places, facts).

```python
entities = {
    "user_name": "Alice",
    "user_preferences": {"units": "celsius"},
    "last_query": "weather in Paris",
    "mentioned_cities": ["Paris", "London"]
}
```

**Pros**: Structured, queryable
**Cons**: Requires extraction logic

### 4. Vector Memory (Semantic)
Stores embeddings for semantic search.

```python
# Store conversation as vectors
memory_db = VectorStore()
memory_db.add("Alice prefers metric units", embedding=[0.1, 0.2, ...])
memory_db.add("Alice visited Paris", embedding=[0.3, 0.1, ...])

# Retrieve relevant memories
query = "What are Alice's preferences?"
relevant = memory_db.search(query, k=3)
```

**Pros**: Scalable, semantic retrieval
**Cons**: Requires embeddings, more complex

## 🔧 Implementation Patterns

### Pattern 1: Buffer Window Memory
```python
class BufferMemory:
    def __init__(self, max_messages: int = 10):
        self.messages = []
        self.max_messages = max_messages

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context(self) -> list:
        return self.messages
```

### Pattern 2: Summary Memory
```python
class SummaryMemory:
    def __init__(self, llm):
        self.llm = llm
        self.summary = ""
        self.recent = []

    def add(self, role: str, content: str):
        self.recent.append({"role": role, "content": content})

        # Summarize every 10 messages
        if len(self.recent) >= 10:
            self.summary = self.llm.summarize(
                self.summary + "\n" + str(self.recent[:8])
            )
            self.recent = self.recent[8:]  # Keep 2 most recent

    def get_context(self) -> str:
        return f"Summary: {self.summary}\n\nRecent: {self.recent}"
```

### Pattern 3: Hybrid Memory
```python
class HybridMemory:
    def __init__(self):
        self.short_term = []  # Recent messages
        self.entities = {}     # Extracted facts
        self.summary = ""      # Conversation summary

    def add(self, role: str, content: str):
        # Add to short-term
        self.short_term.append({"role": role, "content": content})

        # Extract entities (simplified)
        if "my name is" in content.lower():
            name = extract_name(content)
            self.entities["user_name"] = name

    def get_context(self) -> dict:
        return {
            "recent": self.short_term[-5:],
            "entities": self.entities,
            "summary": self.summary
        }
```

## 🚀 Practical Example

See `conversation_agent.py` for a complete agent with memory that:
- Remembers user preferences
- Tracks conversation history
- Maintains entity information
- Provides context-aware responses

## 📊 Memory Management Strategies

### Token Budget Management
```python
def manage_tokens(messages: list, max_tokens: int = 4000):
    """Keep messages within token budget"""
    total_tokens = sum(count_tokens(msg) for msg in messages)

    while total_tokens > max_tokens:
        # Remove oldest non-system message
        for i, msg in enumerate(messages):
            if msg["role"] != "system":
                messages.pop(i)
                break

    return messages
```

### Importance-Based Retention
```python
def prioritize_messages(messages: list):
    """Keep important messages, summarize rest"""
    important = []
    for msg in messages:
        # Keep if contains key information
        if any(keyword in msg["content"].lower()
               for keyword in ["name", "prefer", "important"]):
            important.append(msg)

    return important + messages[-3:]  # + 3 most recent
```

## 🎓 Best Practices

### 1. Clear Memory Structure
```python
# ❌ Unstructured
memory = ["Alice likes Paris", "weather query", "temperature 20C"]

# ✅ Structured
memory = {
    "user_profile": {"name": "Alice", "preferences": {"city": "Paris"}},
    "conversation": [{"query": "weather", "response": "20C"}]
}
```

### 2. Privacy Considerations
```python
# Sanitize sensitive data
def sanitize(message: str) -> str:
    # Remove PII, credit cards, etc.
    import re
    message = re.sub(r'\b\d{16}\b', '[CARD]', message)
    message = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', message)
    return message
```

### 3. Memory Persistence
```python
# Save memory to disk/database
def save_memory(user_id: str, memory: dict):
    with open(f"memory_{user_id}.json", "w") as f:
        json.dump(memory, f)

def load_memory(user_id: str) -> dict:
    try:
        with open(f"memory_{user_id}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
```

## 🔗 Resources

- [LangChain Memory](https://python.langchain.com/docs/modules/memory/)
- [Semantic Kernel Memory](https://learn.microsoft.com/en-us/semantic-kernel/memories/)
- [OpenAI Assistants API](https://platform.openai.com/docs/assistants/overview)
