from ollama import chat

response = chat(
    model="qwen3.5:4b",
    messages=[
        {
            "role": "system",
            "content": "You are a reliable customer support assistant."
        },
        {
            "role": "user",
            "content": "Respond only with the word READY."
        }
    ],
    options={
        "temperature": 0
    }
)

print(response.message.content)