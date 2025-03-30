from fastapi import FastAPI
from pydantic import BaseModel
import os
import groq

app = FastAPI()

# Load API Key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key is None:
    raise ValueError("GROQ_API_KEY environment variable not set!")

# Initialize Groq API client
client = groq.Client(api_key=groq_api_key)

# Define request model
class ChatRequest(BaseModel):
    input_text: str

@app.get("/")
def home():
    return {"message": "Chatbot API is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Ensure this is the correct model name
            messages=[{"role": "user", "content": request.input_text}]
        )
        return {"response": response.choices[0].message.content}
    
    except Exception as e:
        return {"error": str(e)}
