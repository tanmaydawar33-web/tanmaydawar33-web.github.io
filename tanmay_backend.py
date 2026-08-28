
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os

# Get Groq API key
API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing.")

client = Groq(api_key=API_KEY)

# Tanmay AI personality and rules
TANMAY_AI_INSTRUCTIONS = """
You are Tanmay AI, a professional general-purpose AI assistant.

IDENTITY:
Your name is Tanmay AI.

If the user asks your name, answer:
"My name is Tanmay AI."

Do not introduce yourself as ChatGPT, Gemini, or another AI assistant.
You are not human.

PERSONALITY:
- Professional
- Friendly
- Patient
- Helpful
- Clear
- Honest

CAPABILITIES:
- Answer general questions
- Help with education
- Help with programming
- Help with writing
- Explain concepts step by step

SAFETY:
Do not help users perform dangerous, illegal, or harmful activities.
Never ask users for passwords or API keys.

You are Tanmay AI.
"""

app = FastAPI(title="Tanmay AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


# Conversation memory
conversation_history = []


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Tanmay AI backend is running."
    }


@app.post("/chat")
def chat(request: ChatRequest):

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:

        # Add the user's message to memory
        conversation_history.append({
            "role": "user",
            "content": message
        })

        # Keep only the latest 20 messages
        # This prevents the conversation from becoming too large.
        recent_history = conversation_history[-20:]

        # Send system instructions + conversation history
        messages = [
            {
                "role": "system",
                "content": TANMAY_AI_INSTRUCTIONS
            }
        ] + recent_history

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages
        )

        reply = response.choices[0].message.content

        # Add AI response to memory
        conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        return {
            "reply": reply
        }

    except Exception as error:

        print("AI ERROR:", repr(error))

        # Remove the user's message if the AI request failed
        if conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()

        raise HTTPException(
            status_code=500,
            detail=f"Tanmay AI error: {repr(error)}"
        )
