from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from app.chatbot import YouTubeChatbot

# Load environment variables
load_dotenv()

app = FastAPI(title="YouTube Chat API")

# Initialize chatbot with environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")

chatbot = YouTubeChatbot(GROQ_API_KEY)

class ProcessVideoRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to YouTube Chat API"}

@app.post("/process-video")
async def process_video(request: ProcessVideoRequest):
    try:
        result = chatbot.process_url(request.url)
        if result.startswith("❌"):
            raise HTTPException(status_code=400, detail=result)
        return {"status": "success", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = chatbot.chat(request.message)
        if response.startswith("⚠️"):
            raise HTTPException(status_code=400, detail=response)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))