from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

# Import functions from your converted scripts
from base import run_base_chat
from premium import run_premium_chat

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat/base")
def chat_base(req: ChatRequest):
    response = run_base_chat(req.message)
    return {"response": response}

@app.post("/chat/premium")
def chat_premium(req: ChatRequest):
    response = run_premium_chat(req.message)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
