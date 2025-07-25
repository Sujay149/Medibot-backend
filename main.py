from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

# Import function only from base script
from base import run_base_chat

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat/base")
def chat_base(req: ChatRequest):
    response = run_base_chat(req.message)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
