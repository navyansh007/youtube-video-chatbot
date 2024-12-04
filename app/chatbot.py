import groq
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Dict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class YouTubeChatbot:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        self.client = groq.Groq(api_key=api_key)
        self.chat_history = []
        self.transcript_chunks = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

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

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks of approximately equal size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_size += len(word) + 1  # +1 for space
            if current_size > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def process_url(self, url: str) -> str:
        """Process YouTube URL and create TF-IDF vectors."""
        try:
            # Extract video ID
            video_id = self.extract_video_id(url)
            if not video_id:
                return "❌ Error: Invalid YouTube URL format"

            # Get transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            full_transcript = " ".join([entry['text'] for entry in transcript_list])

            # Split into chunks
            self.transcript_chunks = self.chunk_text(full_transcript)
            
            # Create TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.fit_transform(self.transcript_chunks)

            return "✅ Video transcript processed successfully!"
        except Exception as e:
            return f"❌ Error processing video: {str(e)}"

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context using TF-IDF similarity."""
        if not self.tfidf_matrix or not self.transcript_chunks:
            return ""

        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # Get top k chunks
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return "\n".join([self.transcript_chunks[i] for i in top_k_indices])

    def chat(self, message: str) -> str:
        """Process a chat message and return the response."""
        if not self.tfidf_matrix or not self.transcript_chunks:
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
                    """
                }
            ]

            # Add chat history (last 4 messages)
            for msg in self.chat_history[-4:]:
                messages.append(msg)

            # Add context and current message
            messages.append({
                "role": "user",
                "content": f"Context from video transcript:\n{context}\n\nUser question: {message}"
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