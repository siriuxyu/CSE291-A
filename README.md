# React Agent Server

This project provides a FastAPI server for the React Agent, a LangGraph-based agent that can reason and act using tools.

## Features

- **React Agent**: A reasoning and action agent powered by LangGraph
- **Tool Integration**: Built-in tools for weather queries and text translation
- **FastAPI Server**: RESTful API endpoints for agent interactions
- **Docker Support**: Containerized deployment ready
- **Async Support**: Full async/await support for high performance

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

### API Endpoints

#### 1. Full Agent Invocation (`POST /invoke`)
Submit a conversation with multiple messages and get the complete agent response:

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in San Diego?"},
    {"role": "assistant", "content": "Let me check that for you."}
  ],
  "system_prompt": "You are a helpful AI assistant.",
  "model": "anthropic/claude-sonnet-4-5-20250929",
  "max_search_results": 10
}
```

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
For single message interactions:

```json
{
  "message": "Translate 'Hello world' to French",
  "system_prompt": "You are a helpful translator.",
  "model": "anthropic/claude-sonnet-4-5-20250929"
}
```

Response:
```json
{
  "response": "Bonjour le monde"
}
```

#### 3. Health Check (`GET /health`)
Check if the service is running:

```json
{
  "status": "healthy",
  "service": "react-agent"
}
```

### Available Tools

The agent comes with built-in tools:

- **`get_weather`**: Get weather information for a city
- **`translate_text`**: Translate text to any target language

### Docker Deployment

Build and run with Docker:

```bash
docker build -t react-agent-server .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your-key react-agent-server
```

## Project Structure

- `server.py` - FastAPI server for the React Agent
- `graph.py` - Main agent graph definition
- `state.py` - State management for conversations
- `context.py` - Configurable context parameters
- `tools.py` - Available tools for the agent
- `utils.py` - Utility functions
- `run_agent.py` - Direct agent runner (CLI)
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies

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
    "system_prompt": "You are a friendly assistant"
})
print(response.json()["final_response"])
```
