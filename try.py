from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ollama import chat

app = FastAPI()

def generate_stream():
    stream = chat(
        model='mistral',
        messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
        stream=True,
    )
    for chunk in stream:
        yield chunk['message']['content']

@app.get("/stream_chat")
async def stream_chat():
    return StreamingResponse(generate_stream(), media_type="text/plain", headers={"X-Accel-Buffering": "no"})
