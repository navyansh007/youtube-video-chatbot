from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import groq
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
import re

class YouTubeChatbot:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        self.client = groq.Groq(api_key=api_key)
        self.chat_history = []
        self.db = None
        self.embeddings = HuggingFaceEmbeddings()

    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:shorts\/)([0-9A-Za-z_-]{11})',
            r'^([0-9A-Za-z_-]{11})$'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def process_url(self, url: str) -> str:
        """Process YouTube URL and create vector store."""
        try:
            # Extract video ID
            video_id = self.extract_video_id(url)
            if not video_id:
                return "❌ Error: Invalid YouTube URL format"

            # Get transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

            # Combine transcript text
            full_transcript = " ".join([entry['text'] for entry in transcript_list])

            # Create document
            doc = Document(
                page_content=full_transcript,
                metadata={"video_id": video_id, "source": url}
            )

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            texts = text_splitter.split_documents([doc])

            # Create vector store
            self.db = FAISS.from_documents(texts, self.embeddings)

            return f"✅ Video transcript processed successfully!"
        except Exception as e:
            return f"❌ Error processing video: {str(e)}"

    def get_relevant_context(self, query: str) -> str:
        """Retrieve relevant context from the vector store."""
        if not self.db:
            return ""

        docs = self.db.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in docs])

    def chat(self, message: str) -> str:
        """Process a chat message and return the response."""
        if not self.db:
            return "⚠️ Please process a YouTube URL first before chatting."

        try:
            # Get relevant context
            context = self.get_relevant_context(message)

            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": """You are a specialized assistant that ONLY discusses the content from the provided YouTube video transcript.

                    IMPORTANT RULES:
                    1. ONLY answer questions based on the exact information provided in the video transcript context.
                    2. If the question cannot be answered using ONLY the provided transcript context, respond with: "I cannot answer this question as it's not covered in the video content."
                    3. Do not use any external knowledge or make assumptions beyond what's explicitly stated in the transcript.
                    4. If the question is about the video but the relevant information isn't in the current context, say: "While this question is about the video, I don't have access to this specific part of the content in my current context."
                    5. For any off-topic questions not related to the video, respond with: "I can only answer questions about the content of this specific video. Please ask something related to the video."

                    Remember: You are ONLY knowledgeable about the content provided in the transcript context. Do not provide any information beyond this scope."""
                }
            ]

            # Add chat history
            for msg in self.chat_history[-4:]:  # Keep last 4 messages for context window
                messages.append(msg)

            # Add context and current message
            messages.append({
                "role": "user",
                "content": f"""Context from video transcript:\n{context}\n\nUser question: {message}"""
            })

            # Get response from Groq
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="mixtral-8x7b-32768",
                temperature=0.7,
                max_tokens=1024,
            )

            # Extract response
            response = chat_completion.choices[0].message.content

            # Update chat history
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": response})

            return response
        except Exception as e:
            return f"❌ Error processing message: {str(e)}"

app = FastAPI(title="YouTube Chat API")

# Initialize chatbot with environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
chatbot = YouTubeChatbot(GROQ_API_KEY)

class ProcessVideoRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

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

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)