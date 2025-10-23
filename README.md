# React Agent Server

This project provides a FastAPI server for the React Agent, a LangGraph-based agent that can reason and act using tools.

## Features

- **React Agent**: A reasoning and action agent powered by LangGraph
- **Tool Integration**: Built-in tools for calculations, weather queries, translation, web reading, and file system search
- **FastAPI Server**: RESTful API endpoints for agent interactions
- **CLI Interface**: Direct agent runner for testing and development
- **Docker Support**: Containerized deployment ready
- **Async Support**: Full async/await support for high performance
- **State Management**: Conversation history and context tracking

## Quick Start

### Running the React Agent Server

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export ANTHROPIC_API_KEY="your-api-key"     # Required for Claude models
```

3. Run the server:
```bash
python server.py
```

The server will start on `http://localhost:8000`

### Running the Agent Directly (CLI)

For testing and development, you can run the agent directly:

```bash
python run_agent.py
```

This will execute the example queries defined in `run_agent.py`.

### API Endpoints

#### 1. Full Agent Invocation (`POST /invoke`)
Submit a conversation with multiple messages and get the complete agent response:

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in San Diego?"},
    {"role": "assistant", "content": "Let me check that for you."}
  ],
  "userid": "user123",
  "system_prompt": "You are a helpful AI assistant.",
  "model": "anthropic/claude-sonnet-4-5-20250929",
  "max_search_results": 10
}
```

**Parameters:**
- `messages`: List of conversation messages
- `userid`: Required. User ID for maintaining conversation history
- `system_prompt`: Optional. Custom system prompt
- `model`: Optional. Model to use (default: anthropic/claude-sonnet-4-5-20250929)
- `max_search_results`: Optional. Maximum search results (default: 10)

Response:
```json
{
  "messages": [
    {
      "role": "assistant",
      "content": "I'll check the weather in San Diego for you.",
      "tool_calls": [{"name": "get_weather", "args": {"city": "San Diego"}}]
    },
    {
      "role": "tool",
      "content": "Sunny, 75°F"
    },
    {
      "role": "assistant", 
      "content": "The weather in San Diego is sunny with a temperature of 75°F."
    }
  ],
  "final_response": "The weather in San Diego is sunny with a temperature of 75°F."
}
```

#### 2. Simple Chat (`POST /chat`)
For single message interactions without conversation history:

**Request Body:**
```json
{
  "message": "Translate 'Hello world' to French"
}
```

**Parameters:**
- `message`: Required. The user's message

**Response:**
```json
{
  "response": "Bonjour le monde"
}
```

#### 3. Check State (`GET /state/{userid}`)
Check the conversation state for a specific user:

**Response:**
```json
{
  "userid": "user123",
  "state": {...},
  "next_node": "call_model",
  "config": {...}
}
```

**Parameters:**
- `userid`: Path parameter. User ID to check state for

### Available Tools

The agent comes with built-in tools:

- **`calculator`**: Evaluate mathematical expressions safely using basic math functions
- **`get_weather`**: Get weather information for a city
- **`translator`**: Translate text to any target language
- **`web_reader`**: Fetch and extract content from web pages
- **`file_system_search`**: Search for files in the file system

### Docker Deployment

Build and run with Docker:

```bash
docker build -t react-agent-server .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your-key react-agent-server
```

## Project Structure

```
CSE291-A/
├── agent/                      # Core agent implementation
│   ├── __init__.py
│   ├── graph.py               # Main agent graph definition
│   ├── state.py               # State management for conversations
│   ├── context.py             # Configurable context parameters
│   ├── prompts.py             # System prompts
│   └── utils.py               # Utility functions
├── tools/                      # Agent tools
│   ├── __init__.py
│   ├── calculator.py          # Mathematical calculations
│   ├── get_weather.py         # Weather information
│   ├── translator.py          # Text translation
│   ├── web_reader.py          # Web content extraction
│   └── file_system_search.py  # File system operations
├── server.py                   # FastAPI server entry point
├── run_agent.py               # CLI agent runner
├── tests/                      # Test suite
│   ├── __init__.py
│   └── test_translation.py
├── benchmark/                  # Benchmark datasets
│   ├── short.json
│   ├── medium.json
│   ├── long.json
│   └── locomo1.json
├── Dockerfile                  # Container configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Environment Variables

- `ANTHROPIC_API_KEY`: API key for Claude models (required)
- `SYSTEM_PROMPT`: Default system prompt (optional)
- `MODEL`: Default model to use (optional)
- `MAX_SEARCH_RESULTS`: Maximum search results (optional)

## Example Usage

### Using curl:

```bash
# Simple chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Tokyo?"}'

# Full conversation
curl -X POST "http://localhost:8000/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Translate hello to Spanish"}],
    "userid": "test_user",
    "system_prompt": "You are a helpful translator"
  }'
```

### Using Python:

```python
import requests

# Simple chat
response = requests.post("http://localhost:8000/chat", json={
    "message": "What's the weather in Paris?"
})
print(response.json()["response"])

# Full conversation
response = requests.post("http://localhost:8000/invoke", json={
    "messages": [{"role": "user", "content": "Hello"}],
    "userid": "test_user",
    "system_prompt": "You are a friendly assistant"
})
print(response.json()["final_response"])

# Check conversation state
response = requests.get("http://localhost:8000/state/test_user")
print(response.json())
```
