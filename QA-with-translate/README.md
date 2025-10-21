# QA with Translate Agent

A sophisticated question-answering agent that combines question processing with automatic translation capabilities using LangGraph and Anthropic's Claude. The agent follows a two-step workflow: first answering the question, then translating the response to the specified language. The system includes both a Python API and a FastAPI web server for easy integration.

## Features

- **Two-Step Workflow**: First answers questions, then translates responses
- **Multi-Language Support**: Translates to any specified language
- **Honest Responses**: Configured to admit when it doesn't know answers
- **LangGraph Integration**: Uses LangGraph for structured workflow management
- **Claude AI**: Powered by Anthropic's Claude-3-5-Haiku model
- **State Management**: Supports Redis-based persistence or in-memory storage
- **Thread-based Sessions**: Each user session maintains conversation state
- **FastAPI Web Server**: RESTful API endpoints for easy integration
- **Type Safety**: Full TypeScript-style type annotations with TypedDict

## Prerequisites

- Python 3.8 or higher
- Anthropic API account with access to Claude models
- Internet connection for API calls
- Redis server (optional, for persistent state storage)

## Environment Variables

Create a `.env` file in the project directory with the following variable:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
REDIS_URL=redis://localhost:6379  # Optional: for persistent state storage
```

### API Key Setup

1. **Anthropic API Key**: 
   - Sign up at [Anthropic Console](https://console.anthropic.com/)
   - Navigate to API Keys section
   - Create a new API key
   - Ensure you have access to Claude-3-5-Haiku model

2. **Redis Setup (Optional)**:
   - Install Redis server locally or use a cloud Redis service
   - If Redis is not available, the system will fall back to in-memory storage
   - Note: In-memory storage is not persistent across server restarts

## Installation

1. Clone or navigate to the QA-with-translate directory:
```bash
cd QA-with-translate
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (see Environment Variables section above)

## Usage

### Web Server Usage (Recommended)

Start the FastAPI server:

```bash
python server.py
```

The server will start on `http://localhost:8000`. You can interact with it using HTTP requests:

#### Ask a Question
```bash
curl -X POST "http://localhost:8000/invoke" \
     -H "Content-Type: application/json" \
     -d '{
       "input_question": "What is the capital of France?",
       "thread_id": "user123",
       "language": "French"
     }'
```

#### Check Thread State
```bash
curl "http://localhost:8000/state/user123"
```

#### View All States
```bash
curl "http://localhost:8000/states"
```

### Direct Python Usage

Import and use the agent in your code:

```python
from QA_with_Trans import app

# Ask a question and get answer in French (default)
# Note: thread_id is required for state management
response = app("What is the capital of France?", "user123", "French")
print(response)

# Ask a question and get answer in Spanish
response = app("What is machine learning?", "user123", "Spanish")
print(response)

# Ask a question and get answer in German
response = app("How does photosynthesis work?", "user123", "German")
print(response)
```

### Interactive Usage

Create a simple interactive script:

```python
from QA_with_Trans import app
import uuid

def interactive_qa():
    # Generate a unique thread ID for this session
    thread_id = str(uuid.uuid4())
    print(f"Session ID: {thread_id}")
    
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        language = input("Enter target language (default: French): ").strip()
        if not language:
            language = "French"
        
        try:
            response = app(question, thread_id, language)
            print(f"\nAnswer in {language}: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    interactive_qa()
```

### Programmatic Integration

Use the agent as part of a larger application:

```python
from QA_with_Trans import app
import uuid

def process_multilingual_questions(questions, languages):
    results = {}
    thread_id = str(uuid.uuid4())  # Generate unique thread ID
    
    for question in questions:
        for language in languages:
            key = f"{question}_{language}"
            results[key] = app(question, thread_id, language)
    return results

# Example usage
questions = ["What is AI?", "How do computers work?"]
languages = ["French", "Spanish", "German"]
results = process_multilingual_questions(questions, languages)
```

## API Endpoints

When running the FastAPI server, the following endpoints are available:

- `POST /invoke` - Ask a question and get a translated answer
- `GET /state/{thread_id}` - Get the current state of a specific thread
- `GET /states` - View all saved states (for debugging)

### Request/Response Format

#### POST /invoke
```json
{
  "input_question": "What is machine learning?",
  "thread_id": "user123",
  "language": "Spanish"
}
```

Response:
```json
{
  "answer": "El aprendizaje automático es un campo de la inteligencia artificial..."
}
```

#### GET /state/{thread_id}
Response:
```json
{
  "thread_id": "user123",
  "state": {
    "input_text": "What is machine learning?",
    "language": "Spanish",
    "step1_output": "Machine learning is a field of artificial intelligence...",
    "output_text": "El aprendizaje automático es un campo de la inteligencia artificial..."
  },
  "next_node": ["llm_1"],
  "config": {...}
}
```

## Customization

### Model Configuration

Change the Claude model used by modifying the model parameter:

```python
# In the main script, change the model
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")  # Use Sonnet instead of Haiku
```

### Language Customization

Modify the default language in the `app` function:

```python
def app(input_question: str, thread_id: str, language: str = "Spanish") -> str:  # Change default to Spanish
    # ... rest of the code
```

### Prompt Customization

Customize the prompts used in each step:

```python
def llm_1(state: State) -> State:
    # Customize the question-answering prompt
    prompt = f"Please provide a comprehensive answer to: {state['input_text']}. Be thorough and accurate."
    llm_response = llm.invoke([{"role": "user", "content": prompt}])
    return {"step1_output": llm_response.content, "output_text": ""}

def llm_2(state: State) -> State:
    # Customize the translation prompt
    prompt = f"Translate the following text to {state['language']} while maintaining the original meaning and tone: {state['step1_output']}"
    llm_response = llm.invoke([{"role": "user", "content": prompt}])
    return {"output_text": llm_response.content}
```

### State Management

Extend the State TypedDict to include additional fields:

```python
class State(TypedDict):
    # Input
    input_text: str
    language: str
    
    # Intermediate
    step1_output: str
    confidence_score: float  # Add confidence scoring
    
    # Output
    output_text: str
    translation_quality: str  # Add translation quality assessment
```

### Graph Customization

Add additional nodes to the workflow:

```python
def build_graph():
    graph_builder = StateGraph(State)
    
    graph_builder.add_node("llm_1", llm_1)
    graph_builder.add_node("llm_2", llm_2)
    graph_builder.add_node("quality_check", quality_check)  # Add quality check node
    
    graph_builder.add_edge(START, "llm_1")
    graph_builder.add_edge("llm_1", "llm_2")
    graph_builder.add_edge("llm_2", "quality_check")
    graph_builder.add_edge("quality_check", "llm_1")  # Loop back for continuous conversation
    
    graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["quality_check"]
    )
    return graph
```

### Redis Configuration

Configure Redis for persistent state storage:

```python
# Set environment variable
import os
os.environ["REDIS_URL"] = "redis://localhost:6379"

# Or use Redis with authentication
os.environ["REDIS_URL"] = "redis://username:password@localhost:6379"
```

### Docker Support

The project includes a Dockerfile for containerized deployment:

```bash
# Build the Docker image
docker build -t qa-translate-agent .

# Run the container
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your_key qa-translate-agent
```

## Architecture

The system uses LangGraph to create a stateful conversation flow:

1. **Input Processing**: User question and target language are received
2. **Question Answering**: Claude answers the question honestly
3. **Translation**: The answer is translated to the target language
4. **State Persistence**: Conversation state is saved for future interactions
5. **Loop Continuation**: The system is ready for the next question

The graph uses interrupt points to save state after each complete question-answer-translation cycle, allowing for continuous conversations while maintaining context.

## Troubleshooting

### Common Issues

1. **Redis Connection Error**: If Redis is not available, the system will automatically fall back to in-memory storage
2. **API Key Issues**: Ensure your Anthropic API key is correctly set in the environment variables
3. **Thread ID Management**: Each conversation requires a unique thread_id for proper state management

### Debugging

Use the built-in debugging functions:

```python
from QA_with_Trans import view_thread_state, view_all_states

# View state of a specific thread
view_thread_state("user123")

# View all saved states
view_all_states()
```
