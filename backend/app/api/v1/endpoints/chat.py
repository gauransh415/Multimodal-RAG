from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import json
import base64

from app.services.rag_service import enhanced_rag_service, EnhancedRAGResponse
from app.models.chat import ChatSession, ChatMessage
from app.core.database import get_db
from sqlalchemy.orm import Session

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    include_sources: bool = True
    include_reasoning: bool = True

class MultimodalChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    image_data: Optional[List[str]] = None  # Base64 encoded images
    include_sources: bool = True
    include_reasoning: bool = True

class ChatResponse(BaseModel):
    message: str
    session_id: str
    message_id: str
    sources: List[Dict[str, Any]] = []
    confidence: float = 0.0
    reasoning: Optional[str] = None
    document_types: List[str] = []

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """Handle text-only chat requests with RAG"""
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get session context for conversation history
        session_context = await _get_session_context(session_id, db)
        
        # Process RAG query
        rag_response = await enhanced_rag_service.multimodal_query(
            query=request.message,
            session_context=session_context
        )
        
        # Save conversation to database
        message_id = str(uuid.uuid4())
        await _save_conversation(
            session_id=session_id,
            user_message=request.message,
            assistant_message=rag_response.answer,
            message_id=message_id,
            db=db
        )
        
        return ChatResponse(
            message=rag_response.answer,
            session_id=session_id,
            message_id=message_id,
            sources=rag_response.sources if request.include_sources else [],
            confidence=rag_response.confidence,
            reasoning=rag_response.reasoning if request.include_reasoning else None,
            document_types=rag_response.document_types
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/multimodal", response_model=ChatResponse)
async def multimodal_chat_endpoint(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    include_sources: bool = Form(True),
    include_reasoning: bool = Form(True),
    images: List[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """Handle multimodal chat requests (text + images) with RAG"""
    
    try:
        # Generate session ID if not provided
        session_id = session_id or str(uuid.uuid4())
        
        # Process uploaded images
        image_data_list = []
        if images:
            for image in images:
                if image.content_type.startswith('image/'):
                    image_bytes = await image.read()
                    image_data_list.append(image_bytes)
        
        # Get session context
        session_context = await _get_session_context(session_id, db)
        
        # Process multimodal RAG query
        rag_response = await enhanced_rag_service.multimodal_query(
            query=message,
            images=image_data_list,
            session_context=session_context
        )
        
        # Save conversation to database
        message_id = str(uuid.uuid4())
        await _save_conversation(
            session_id=session_id,
            user_message=message,
            assistant_message=rag_response.answer,
            message_id=message_id,
            db=db,
            has_images=bool(image_data_list)
        )
        
        return ChatResponse(
            message=rag_response.answer,
            session_id=session_id,
            message_id=message_id,
            sources=rag_response.sources if include_sources else [],
            confidence=rag_response.confidence,
            reasoning=rag_response.reasoning if include_reasoning else None,
            document_types=rag_response.document_types
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/sessions")
async def get_chat_sessions(
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get user's chat sessions"""
    
    query = db.query(ChatSession)
    if user_id:
        query = query.filter(ChatSession.user_id == user_id)
    
    sessions = query.order_by(ChatSession.updated_at.desc()).all()
    
    return {
        "sessions": [
            {
                "id": session.id,
                "session_id": session.session_id,
                "title": session.title,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "message_count": len(session.messages)
            }
            for session in sessions
        ]
    }

@router.get("/chat/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get messages for a specific chat session"""
    
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session.id
    ).order_by(ChatMessage.timestamp).all()
    
    return {
        "session_id": session_id,
        "messages": [
            {
                "message_id": msg.message_id,
                "role": msg.role,
                "content": msg.content,
                "content_type": msg.content_type,
                "timestamp": msg.timestamp,
                "metadata": json.loads(msg.metadata) if msg.metadata else {}
            }
            for msg in messages
        ]
    }

@router.delete("/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Delete a chat session and all its messages"""
    
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Delete all messages first
        db.query(ChatMessage).filter(ChatMessage.session_id == session.id).delete()
        
        # Delete session
        db.delete(session)
        db.commit()
        
        return {"message": "Session deleted successfully", "session_id": session_id}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def _get_session_context(session_id: str, db: Session) -> List[Dict]:
    """Get recent conversation context for a session"""
    
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        return []
    
    # Get last 6 messages (3 exchanges)
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session.id
    ).order_by(ChatMessage.timestamp.desc()).limit(6).all()
    
    # Group messages into exchanges
    context = []
    current_exchange = {}
    
    for msg in reversed(messages):  # Reverse to get chronological order
        if msg.role == "user":
            if current_exchange:
                context.append(current_exchange)
            current_exchange = {"user": msg.content}
        elif msg.role == "assistant":
            current_exchange["assistant"] = msg.content
    
    if current_exchange:
        context.append(current_exchange)
    
    return context[-3:]  # Return last 3 exchanges

async def _save_conversation(
    session_id: str,
    user_message: str,
    assistant_message: str,
    message_id: str,
    db: Session,
    has_images: bool = False
):
    """Save conversation to database"""
    
    # Get or create session
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        session = ChatSession(
            session_id=session_id,
            title=user_message[:50] + "..." if len(user_message) > 50 else user_message
        )
        db.add(session)
        db.commit()
        db.refresh(session)
    
    # Save user message
    user_msg = ChatMessage(
        session_id=session.id,
        message_id=f"{message_id}_user",
        role="user",
        content=user_message,
        content_type="multimodal" if has_images else "text",
        metadata=json.dumps({"has_images": has_images}) if has_images else None
    )
    db.add(user_msg)
    
    # Save assistant message
    assistant_msg = ChatMessage(
        session_id=session.id,
        message_id=f"{message_id}_assistant",
        role="assistant",
        content=assistant_message,
        content_type="text"
    )
    db.add(assistant_msg)
    
    # Update session timestamp
    session.updated_at = func.now()
    
    db.commit()
