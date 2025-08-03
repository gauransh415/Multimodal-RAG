# api/v1/endpoints/chat.py
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import json

from app.services.rag_service import rag_service, RAGResponse
from app.models.chat import ChatSession, ChatMessage
from app.core.database import get_db
from sqlalchemy.orm import Session

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    include_sources: bool = True

class ChatResponse(BaseModel):
    message: str
    session_id: str
    message_id: str
    sources: List[Dict[str, Any]] = []
    confidence: float = 0.0

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """Handle chat requests with RAG"""
    try:
      response = rag_service.chat(request.message, request.session_id, request.include_sources, db)
      return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      
