# React Agent Server

This project provides a FastAPI server for the React Agent, a LangGraph-based agent that can reason and act using tools.

## Features

- **React Agent**: A reasoning and action agent powered by LangGraph
- **Tool Integration**: Built-in tools for calculations, weather queries, translation, web reading, and file system search
- **FastAPI Server**: RESTful API endpoints for agent interactions
- **Session Management**: Persistent session tracking with automatic timeout ([docs](docs/session-management.md))
- **Memory System**: Long-term user preference extraction and context enhancement
- **CLI Interface**: Direct agent runner for testing and development
- **Docker Support**: Containerized deployment ready
- **Async Support**: Full async/await support for high performance
- **Logging**: Automatic logging to both console and file for debugging and monitoring

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

This provides an interactive Q&A interface. Type your questions and the agent will respond using available tools.

#### Running Archived Agent Versions

For comparison and benchmarking purposes, you can also run archived agent versions:

```bash
# Run Phase 1 archived agent
python agent-phase1/run_agent.py

# With debug mode
python agent-phase1/run_agent.py --debug
```

See [agent-phase1/README.md](agent-phase1/README.md) for more details about the archived version.

### API Endpoints

#### 1. Full Agent Invocation (`POST /invoke`)
Submit a conversation with multiple messages and get the complete agent response:

**Request Body:**
```json
{
  "messages": [{"role": "user", "content": "What's the weather in San Diego?"}],
  "userid": "user123",
  "session_id": "user123_a1b2c3d4",
  "force_new_session": false,
  "system_prompt": "You are a helpful AI assistant.",
  "model": "anthropic/claude-sonnet-4-5-20250929",
  "max_search_results": 10
}
```

**Parameters:**
- `messages`: List of conversation messages
- `userid`: Required. User identifier
- `session_id`: Optional. Session ID to continue existing conversation
- `force_new_session`: Optional. Force create new session (default: false)
- `system_prompt`: Optional. Custom system prompt
- `model`: Optional. Model to use (default: anthropic/claude-sonnet-4-5-20250929)
- `max_search_results`: Optional. Maximum search results (default: 10)

**Response:**
```json
{
  "user_id": "user123",
  "session_id": "user123_a1b2c3d4",
  "messages": [...],
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

#### 3. Get User Sessions (`GET /sessions/{userid}`)
Get all sessions for a user. See [Session Management](docs/session-management.md) for details.

**Response:**
```json
{
  "userid": "user123",
  "total_sessions": 3,
  "active_sessions": 1,
  "sessions": [
    {
      "session_id": "user123_a1b2c3d4",
      "last_activity": "2025-01-15T10:30:00",
      "is_active": true,
      "minutes_since_activity": 5
    }
  ]
}
```

#### 4. Check Session State (`GET /state/{session_id}`)
Check the conversation state for a specific session:

**Response:**
```json
{
  "session_id": "user123_a1b2c3d4",
  "state": {...},
  "next_node": "call_model",
  "config": {...}
}
```

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
│   ├── interfaces/            # Interface definitions
│   │   ├── memory_interface.py
│   │   ├── storage_interface.py
│   │   ├── extraction_interface.py
│   │   └── README.md          # Interface documentation
│   ├── memory/                # Memory management
│   │   └── manager.py         # Context memory manager
│   ├── extraction/            # Preference extraction
│   │   ├── preference.py      # Extraction logic
│   │   ├── schemas.py         # Data models
│   │   └── prompts.py         # Extraction prompts
│   ├── graph.py               # Main agent graph
│   ├── state.py               # State management
│   ├── context.py             # Context configuration
│   ├── prompts.py             # System prompts
│   └── utils.py               # Utilities
├── tools/                      # Agent tools
│   ├── calculator.py
│   ├── get_weather.py
│   ├── translator.py
│   ├── web_reader.py
│   └── file_system_search.py
├── docs/                       # Documentation
│   └── session-management.md  # Session management guide
├── tests/                      # Test suite
├── benchmark/                  # Benchmark datasets
├── logs/                       # Application logs
├── server.py                   # FastAPI server
├── run_agent.py               # CLI runner
├── Dockerfile                  # Container config
└── README.md                   # This file
```

## Environment Variables

- `ANTHROPIC_API_KEY`: API key for Claude models (required)
- `SESSION_TIMEOUT_MINUTES`: Session timeout in minutes (default: 30)
- `SYSTEM_PROMPT`: Default system prompt (optional)
- `MODEL`: Default model to use (optional)
- `MAX_SEARCH_RESULTS`: Maximum search results (optional)

## Logging

The application automatically logs to both console and files with a comprehensive structured logging system:

### Log Outputs

- **Console**: Shows INFO level and above logs with simplified format for readability
- **File**: Logs all DEBUG and above logs to `logs/react_agent_YYYYMMDD_HHMMSS.log` in JSON format

### Structured Logging Features

The logging system uses Python's built-in `logging` module with custom JSON formatting:

- **Request Tracking**: Each request gets a unique request ID for tracing through the system
- **User Context**: User IDs are tracked throughout the request lifecycle
- **Function Tracking**: Logs include function names for debugging
- **Performance Metrics**: Duration tracking for operations (in milliseconds)
- **Detailed Context**: Rich details about operations, tool calls, and model interactions
- **Automatic Timestamps**: ISO format timestamps for all log entries
- **Error Tracking**: Full stack traces for exceptions

### Log Levels

- **DEBUG**: Detailed diagnostic information for development
- **INFO**: General operational information (default level)
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages with full stack traces

### Example Log Entry

```json
{
  "timestamp": "2025-10-23T14:40:25.123456",
  "level": "INFO",
  "logger": "agent.graph",
  "message": "Model response received",
  "function": "call_model",
  "duration_ms": 1234.56,
  "details": {
    "has_tool_calls": true,
    "tool_calls_count": 2,
    "response_length": 156
  }
}
```

### What Gets Logged

**API Requests:**
- Request received with message count and user ID
- Context creation (model, system prompt, etc.)
- Agent execution start
- Chunk processing (model calls, tool executions)
- Request completion with duration and statistics

**Agent Operations:**
- Model calls with timing and response details
- Tool calls detection and routing
- Graph execution flow
- State transitions

**System Events:**
- Logging initialization
- Server startup
- Redis connection status
- Error occurrences with full context

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
