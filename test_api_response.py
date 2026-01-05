import requests
import json

# Define the endpoint (vLLM uses OpenAI-compatible paths)
url = "http://localhost:8002/v1/chat/completions"

# Define the payload
payload = {
    "model": "Qwen/Qwen3-0.6B",  # Must match the model name in Step 1
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum physics in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 100
}

# Send the POST request
response = requests.post(url, json=payload)

# Check if request was successful
if response.status_code == 200:
    # .json() converts the response to a Python dictionary
    full_response = response.json()
    
    print("--- FULL RAW RESPONSE ---")
    # json.dumps with indent=4 prints the entire JSON tree clearly
    print(json.dumps(full_response, indent=4))
else:
    print(f"Error: {response.status_code}")
    print(response.text)