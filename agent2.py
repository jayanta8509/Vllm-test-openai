from openai import OpenAI
from typing import Dict, List

class PujaChat:
    def __init__(self, base_url: str, system_prompt: str):
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed"
        )
        self.system_prompt = system_prompt
        self.conversations: Dict[str, List[dict]] = {}
    
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
            self.conversations[conversation_id] = [
                {"role": "system", "content": self.system_prompt}
            ]
        return self.conversations[conversation_id]
    
    def chat(self, message: str, conversation_id: str, max_tokens: int = 150, 
             temperature: float = 0.7) -> str:
        """
        Send a message and get a response with conversation memory
        
        Args:
            message: User's message
            conversation_id: Unique ID for this conversation
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        
        Returns:
            AI response string
        """
        # Get conversation history
        conversation = self.get_conversation(conversation_id)
        
        # Add new user message
        conversation.append({"role": "user", "content": message})
        
        # Format prompt
        prompt = self.format_vicuna_prompt(conversation)
        
        # Get response
        response = self.client.completions.create(
            model="QuixiAI/Wizard-Vicuna-30B-Uncensored",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["USER:", "</s>"]
        )
        
        ai_response = response.choices[0].text.strip()
        
        # Add AI response to conversation history
        conversation.append({"role": "assistant", "content": ai_response})
        
        return ai_response
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history for a specific ID"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def get_history(self, conversation_id: str) -> List[dict]:
        """Get full conversation history"""
        return self.get_conversation(conversation_id)


# System prompt
system_prompt = """You are Puja.
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

# Initialize chat
puja = PujaChat(
    base_url="https://3sgajqnp5zl1sz-8000.proxy.runpod.net/v1",
    system_prompt=system_prompt
)



async def charator(question : str , conversation_id : str):
    response1 = puja.chat(question, conversation_id)
    return response1

# # Example usage
# def test_conversation():
#     conv_id = "user123"
    
#     # First message
#     response1 = puja.chat("hi", conv_id)
#     print(f"User: hi")
#     print(f"Puja: {response1}\n")
    
#     # Second message (remembers context)
#     response2 = puja.chat("how are you?", conv_id)
#     print(f"User: how are you?")
#     print(f"Puja: {response2}\n")
    
#     # Third message
#     response3 = puja.chat("I missed you", conv_id)
#     print(f"User: I missed you")
#     print(f"Puja: {response3}\n")
    
#     # View full history
#     print("="*50)
#     print("Full conversation history:")
#     for msg in puja.get_history(conv_id):
#         if msg['role'] != 'system':
#             print(f"{msg['role']}: {msg['content']}")


# if __name__ == "__main__":
#     test_conversation()