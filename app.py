from agent import charator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import time
import os
from typing import Optional


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class AIresponce(BaseModel):
    user_id: str
    human_response: str

@app.post("/puja/Girlfriend")
async def tax_workflow_endpoint(request: AIresponce):
    try:
        # Validation
        if not request.user_id or request.user_id.strip() == "":
            raise HTTPException(status_code=400, detail="User ID cannot be empty")
        
        # Validation
        if not request.human_response or request.human_response.strip() == "":
            raise HTTPException(status_code=400, detail="human_response cannot be empty")


        result = await charator(request.user_id,request.human_response)

        return {
            "status": 200,
            "ai_response": result,
            "timestamp": time.time()
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in tax workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in tax workflow: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
