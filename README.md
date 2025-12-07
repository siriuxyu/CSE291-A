# React Agent Server

This project provides a FastAPI server for the React Agent, a LangGraph-based agent that can reason and act using tools.

## Features

- **React Agent**: A reasoning and action agent powered by LangGraph
- **Tool Integration**: Built-in tools for calculations, weather queries, translation, web reading, and file system search
- **Claude Web Search**: Native web search integration using Claude's built-in search tool (when using Anthropic models)
- **Memory System**: Long-term memory management with ChromaDB and user-specific namespaces
- **FastAPI Server**: RESTful API endpoints for agent interactions
- **CLI Interface**: Direct agent runner for testing and development
- **Docker Support**: Containerized deployment ready
- **Async Support**: Full async/await support for high performance
- **State Management**: Conversation history and context tracking
- **Persistent Storage**: ChromaDB vector database for long-term memory
- **Logging**: Automatic logging to both console and file for debugging and monitoring

## Quick Start

### Running the React Agent Server

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (or create a `.env` file):
```bash
# Required API keys
export ANTHROPIC_API_KEY="your-api-key"     # Required for Claude models
export OPENAI_API_KEY="your-openai-key"     # Required for LangMem embeddings

# Optional: ChromaDB storage path (default: ./chroma_db_data)
export CHROMA_PERSIST_PATH="./chroma_db_data"
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

#### 4. Reset Session (`POST /reset/{userid}`)
Reset a user's short-term conversation history:

**Parameters:**
- `userid`: Path parameter. User ID to reset
- `preserve_memory`: Query parameter (default: true). If false, also clears long-term memories

**Response:**
```json
{
  "status": "success",
  "userid": "user123",
  "messages_cleared": 5,
  "memory_cleared": false,
  "message": "Session reset for user user123."
}
```

#### 5. Memory Management Endpoints

**List Memories (`GET /memory/{userid}`):**
```bash
curl "http://localhost:8000/memory/user123?limit=100"
```

**Search Memories (`POST /memory/{userid}/search`):**
```bash
curl -X POST "http://localhost:8000/memory/user123/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "favorite food", "limit": 10}'
```

**Store Memory (`POST /memory/{userid}/store`):**
```bash
curl -X POST "http://localhost:8000/memory/user123/store" \
  -H "Content-Type: application/json" \
  -d '{"key": "preference_001", "content": "User prefers dark mode"}'
```

**Delete Memory (`DELETE /memory/{userid}/{key}`):**
```bash
curl -X DELETE "http://localhost:8000/memory/user123/preference_001"
```

**Clear All Memories (`DELETE /memory/{userid}`):**
```bash
curl -X DELETE "http://localhost:8000/memory/user123"
```

### Available Tools

The agent comes with built-in tools:

- **`calculator`**: Evaluate mathematical expressions safely using basic math functions
- **`get_weather`**: Get weather information for a city
- **`translator`**: Translate text to any target language
- **`web_reader`**: Fetch and extract content from web pages
- **`file_system_search`**: Search for files in the file system

**Web Search:**
- **`web_search`** (Claude built-in, default): When using Anthropic models with `enable_web_search=True` (default), Claude uses its own built-in web search tool (`web_search_20250305`). This provides encrypted search results that Claude can process directly, with automatic citation support. The search results are encrypted and only Claude can decrypt them, ensuring secure and accurate information retrieval.
- **`web_searcher`** (fallback): A custom DuckDuckGo-based search tool available as a fallback for non-Anthropic models or when Claude's web search is disabled. This tool returns plain text search results that can be used with the `web_reader` tool to fetch page content.

**Note:** When using Claude models, the built-in `web_search` tool is preferred because it provides better integration, automatic citations, and encrypted results that Claude can process natively. The `web_searcher` tool is kept for compatibility with non-Anthropic models.

**Memory Tools (automatically added per user):**
- **`search_memory`**: Search ChromaDB for previously stored memories
- **`store_memory`**: Store important information for long-term recall

### Docker Deployment

Build and run with Docker:

```bash
docker build -t react-agent-server .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your-key react-agent-server
```

## Project Structure

```
CSE291-A/
├── agent/                      # Core agent implementation (active)
│   ├── __init__.py
│   ├── graph.py               # Main agent graph definition
│   ├── state.py               # State management for conversations
│   ├── context.py             # Configurable context parameters
│   ├── prompts.py             # System prompts
│   ├── utils.py               # Utility functions
│   ├── memory/                # LangMem integration
│   │   ├── __init__.py
│   │   └── langmem_adapter.py # LangMem manager with ChromaDB backend
│   └── storage/               # Persistent storage backends
│       ├── __init__.py
│       ├── vector_storage.py  # ChromaDB vector storage
│       └── embedding_service.py # OpenAI embeddings
├── agent-phase1/               # Phase 1 agent archive (for comparison)
│   ├── graph.py               # Archived agent graph
│   ├── context.py             # Archived context
│   ├── state.py               # Archived state
│   ├── utils.py               # Archived utils
│   ├── prompts.py             # Archived prompts
│   ├── run_agent.py           # Archived CLI runner
│   └── README.md              # Archive documentation
├── tools/                      # Agent tools (shared)
│   ├── __init__.py
│   ├── calculator.py          # Mathematical calculations
│   ├── get_weather.py         # Weather information
│   ├── translator.py          # Text translation
│   ├── web_reader.py          # Web content extraction
│   ├── web_searcher.py        # DuckDuckGo-based web search (fallback)
│   └── file_system_search.py  # File system operations
├── server.py                   # FastAPI server entry point
├── run_agent.py               # CLI agent runner (active)
├── benchmark_runner.py        # Base benchmark utilities
├── locomo_batch_runner.py     # LoCoMo batch benchmark (short-term memory)
├── locomo_memory_runner.py    # LoCoMo memory benchmark (long-term memory)
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_translation.py
│   ├── test_server.py
│   └── test_cases.py
├── benchmark/                  # Benchmark datasets
│   ├── short.json
│   ├── medium.json
│   ├── long.json
│   ├── locomo1.json
│   └── locomo1_converted.json # Converted LoCoMo format
├── chroma_db_data/            # ChromaDB persistent storage
├── logs/                       # Application logs
├── Report/                     # Project reports
├── Dockerfile                  # Container configuration
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in git)
└── README.md                   # This file
```

## Environment Variables

Create a `.env` file in the project root (automatically loaded by `python-dotenv`):

```bash
# Required API Keys
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key  # Required for embeddings

# Optional: ChromaDB Storage Path
CHROMA_PERSIST_PATH=./chroma_db_data
```

### Required
- `ANTHROPIC_API_KEY`: API key for Claude models
- `OPENAI_API_KEY`: API key for OpenAI (required for LangMem embeddings)

### Memory System Configuration
- `LANGMEM_ENABLED`: Enable/disable memory tools (default: "1")

### Storage Backend (ChromaDB)
- `CHROMA_PERSIST_PATH`: ChromaDB storage directory (default: "./chroma_db_data")

### Optional
- `SYSTEM_PROMPT`: Default system prompt
- `MODEL`: Default model to use
- `MAX_SEARCH_RESULTS`: Maximum search results (optional)
- `ENABLE_WEB_SEARCH`: Enable Claude's built-in web search (default: "1" or True)
- `WEB_SEARCH_MAX_USES`: Maximum number of web searches per request (default: 5)
- `REDIS_URL`: Redis connection URL for persistent checkpoints

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

# Search user memories
response = requests.post("http://localhost:8000/memory/test_user/search", json={
    "query": "preferences",
    "limit": 5
})
print(response.json())

# Reset session (keeping memories)
response = requests.post("http://localhost:8000/reset/test_user?preserve_memory=true")
print(response.json())
```

## Memory System

The agent has a unified long-term memory system using ChromaDB. Each user has their own isolated memory namespace.

### How It Works

1. **Memory Tools**: The agent automatically gets `search_memory` and `store_memory` tools for each user
2. **User Isolation**: Memories are stored per-user in ChromaDB
3. **Unified Storage**: Both API endpoints (`/memory/*`) and agent tools use the same ChromaDB backend
4. **Context Recall**: The agent can search past memories when answering questions
5. **Persistent Storage**: All memories persist across server restarts

### Memory Tools

The agent has access to these tools when `user_id` is provided:

| Tool | Description |
|------|-------------|
| `search_memory(query)` | Search ChromaDB for relevant memories matching the query |
| `store_memory(content)` | Store new information in ChromaDB for future recall |

### Storage Architecture

```
┌─────────────────────────┐
│   /memory/* API         │──────┐
│   (store, search, list) │      │
└─────────────────────────┘      │
                                 ▼
                    ┌────────────────────────┐
                    │       ChromaDB         │
                    │   (Persistent Vector   │
                    │      Database)         │
                    │  ./chroma_db_data/     │
                    └────────────────────────┘
                                 ▲
┌─────────────────────────┐      │
│   Agent Memory Tools    │──────┘
│   (search_memory,       │
│    store_memory)        │
└─────────────────────────┘
```

The memory system uses:
- **VectorStorageBackend** (`storage/vector_storage.py`): ChromaDB-based vector database
- **EmbeddingService** (`storage/embedding_service.py`): OpenAI embeddings for semantic search

**Features:**
- Persistent vector storage in `./chroma_db_data/`
- Automatic embedding generation via OpenAI
- Semantic similarity search
- Data persists across server restarts

```bash
# Configure ChromaDB storage path (optional)
export CHROMA_PERSIST_PATH="./chroma_db_data"
```

### Running the LoCoMo Benchmark

Two benchmark runners are available for testing cross-session memory:

#### 1. Batch Runner (`locomo_batch_runner.py`)

Stores conversation transcripts in short-term memory and tests QA recall:

```bash
# Run with default settings
python locomo_batch_runner.py --limit-personas 2

# With custom token limits (to avoid rate limits)
python locomo_batch_runner.py \
  --locomo-file benchmark/locomo1_converted.json \
  --server-url http://localhost:8000 \
  --limit-personas 5 \
  --max-transcript-chars 6000 \
  --max-qa-chars 4000 \
  --max-questions 10 \
  --delay-sec 3

# Results saved to locomo_batch_results.json
```

#### 2. Memory Runner (`locomo_memory_runner.py`)

Tests true cross-session memory by storing conversations in long-term memory:

```bash
# Run memory benchmark
python locomo_memory_runner.py --limit-personas 2

# With custom settings
python locomo_memory_runner.py \
  --limit-personas 5 \
  --max-content-chars 2000 \
  --max-qa-chars 4000 \
  --max-questions 10 \
  --delay-sec 3

# Results saved to locomo_memory_results.json
```

#### Token Limit Parameters

Both runners support token limiting to avoid Anthropic rate limits (30K input tokens/min for Sonnet/Opus):

| Parameter | Batch Runner | Memory Runner | Description |
|-----------|-------------|---------------|-------------|
| `--max-transcript-chars` | 6000 | - | Max chars for conversation transcript |
| `--max-content-chars` | - | 2000 | Max chars per memory item |
| `--max-qa-chars` | 4000 | 4000 | Max chars for QA prompt |
| `--max-questions` | 10 | 10 | Max questions per persona |
| `--delay-sec` | 3.0 | 3.0 | Delay between API calls |

#### Rate Limit Considerations

Claude API rate limits:
- **Sonnet/Opus**: 30K input tokens/min, 50 requests/min
- **Haiku**: 50K input tokens/min, 50 requests/min

With default settings (3s delay, ~1500 tokens/request), you can run ~20 requests/minute while staying within limits.
