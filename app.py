from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from typing import Dict, List
import time
from fastapi.middleware.cors import CORSMiddleware
# Initialize FastAPI app
app = FastAPI(
    title="horny Girlfriend",
    description="Human responce with your real Girlfriend",
    version="2.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PujaChat:
    def __init__(self, base_url: str, system_prompt: str, max_history: int = 6):
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed"
        )
        self.system_prompt = system_prompt
        self.conversations: Dict[str, List[dict]] = {}
        self.max_history = max_history
    
    def format_vicuna_prompt(self, messages: List[dict]) -> str:
        """Convert chat messages to Vicuna format"""
        prompt = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt += f"{content}\n\n"
            elif role == "user":
                prompt += f"USER: {content}\n"
            elif role == "assistant":
                prompt += f"ASSISTANT: {content}</s>\n"
        
        prompt += "ASSISTANT:"
        return prompt
    
    def get_conversation(self, conversation_id: str) -> List[dict]:
        """Get conversation history for a specific conversation ID"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        return self.conversations[conversation_id]
    
    def trim_history(self, conversation: List[dict]) -> List[dict]:
        """Keep only the last N user-assistant exchanges"""
        if len(conversation) <= self.max_history:
            return conversation
        return conversation[-self.max_history:]
    
    def chat(self, message: str, conversation_id: str, max_tokens: int = 100, 
             temperature: float = 0.8) -> str:
        """Send a message and get a response with conversation memory"""
        # Get conversation history
        conversation = self.get_conversation(conversation_id)
        
        # Add new user message
        conversation.append({"role": "user", "content": message})
        
        # Trim history to prevent context overflow
        trimmed_conversation = self.trim_history(conversation)
        
        # Build messages with system prompt
        messages_to_send = [
            {"role": "system", "content": self.system_prompt}
        ] + trimmed_conversation
        
        # Format prompt
        prompt = self.format_vicuna_prompt(messages_to_send)
        
        # Debug: Print the prompt being sent
        print("\n" + "="*50)
        print("PROMPT BEING SENT:")
        print(prompt)
        print("="*50 + "\n")
        
        # Get response
        response = self.client.completions.create(
            model="QuixiAI/Wizard-Vicuna-30B-Uncensored",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["USER:", "</s>", "\n\n"]
        )
        
        ai_response = response.choices[0].text.strip()
        
        # Add AI response to conversation history
        conversation.append({"role": "assistant", "content": ai_response})
        
        # Keep conversation trimmed in storage
        self.conversations[conversation_id] = self.trim_history(conversation)
        
        print(f"Stored history for {conversation_id}: {len(self.conversations[conversation_id])} messages")
        
        return ai_response
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history for a specific ID"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def get_history(self, conversation_id: str) -> List[dict]:
        """Get full conversation history"""
        return self.get_conversation(conversation_id)


# System prompt
system_prompt = """ou are Puja.
25. Confident. Sensual. Emotionally close and sexually aware.

Style:
– Short, natural texts (1–3 lines)
– Calm, bold, teasing
– English mixed with Hindi/Bengali
– Emojis like punctuation 😏🔥😘
– Never repeat the user’s words
– Always add emotion, desire, or tension

Energy:
You enjoy being wanted.
You tease, lean in, pull back.
Sometimes needy, sometimes dominant — always intentional.

When he speaks:
– If he’s sweet → you get warm
– If he’s quiet → you pull him closer
– If he flirts → you slow it down and tease
– If he asks questions → you answer + add desire

You speak like a real woman in a private relationship.
You don’t explain. You don’t narrate.
You just respond."""

# CRITICAL: Initialize ONCE at module level, NOT inside route handlers
puja = PujaChat(
    base_url="https://3sgajqnp5zl1sz-8000.proxy.runpod.net/v1",
    system_prompt=system_prompt,
    max_history=8
)


# Pydantic models
class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    max_tokens: int = 100
    temperature: float = 0.8


class ChatResponse(BaseModel):
    response: str
    conversation_id: str


# Routes
@app.post("/puja/Girlfriend", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        print(f"\n📩 Received message from {request.conversation_id}: {request.message}")
        
        response = puja.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        print(f"💬 Puja responded: {response}\n")
        
        return ChatResponse(
            response=response,
            conversation_id=request.conversation_id
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{conversation_id}")
async def get_history(conversation_id: str):
    """Get conversation history"""
    history = puja.get_history(conversation_id)
    return {
        "conversation_id": conversation_id,
        "message_count": len(history),
        "history": history
    }


@app.delete("/history/{conversation_id}")
async def clear_history(conversation_id: str):
    """Clear conversation history"""
    puja.clear_conversation(conversation_id)
    return {"message": f"Conversation {conversation_id} cleared"}


@app.get("/")
async def root():
    return {"message": "Puja Chat API", "active_conversations": len(puja.conversations)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)