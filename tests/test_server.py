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