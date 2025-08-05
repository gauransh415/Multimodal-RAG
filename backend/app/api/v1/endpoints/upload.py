from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from typing import List, Optional
import os
import uuid
import shutil
from pathlib import Path

from app.core.config import settings
from app.core.database import get_db
from app.services.enhanced_rag_service import enhanced_rag_service
from sqlalchemy.orm import Session

router = APIRouter()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: Optional[int] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload and process a document for RAG"""
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_ext} not supported"
            )
        
        # Check file size
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail="File too large"
            )
        
        # Save file
        file_id = str(uuid.uuid4())
        filename = f"{file_id}{file_ext}"
        file_path = os.path.join(settings.UPLOAD_PATH, filename)
        
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Process document
        document = await enhanced_rag_service.process_and_index_document(
            file_path=file_path,
            filename=file.filename,
            db=db,
            user_id=user_id
        )
        
        return {
            "message": "Document uploaded and processed successfully",
            "document_id": document.id,
            "filename": file.filename,
            "file_type": document.file_type,
            "content_type": document.content_type,
            "chunks_created": len(document.chunks) if document.chunks else 0
        }
        
    except Exception as e:
        # Clean up file if processing failed
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
async def list_documents(
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """List uploaded documents"""
    
    from app.models.document import Document
    
    query = db.query(Document)
    if user_id:
        query = query.filter(Document.user_id == user_id)
    
    documents = query.all()
    
    return {
        "documents": [
            {
                "id": doc.id,
                "filename": doc.original_name,
                "file_type": doc.file_type,
                "content_type": doc.content_type,
                "file_size": doc.file_size,
                "processed": doc.processed,
                "created_at": doc.created_at,
                "chunks_count": len(doc.chunks) if doc.chunks else 0
            }
            for doc in documents
        ]
    }

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Delete a document and remove from index"""
    
    from app.models.document import Document
    
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Remove from vector store
        removed_chunks = enhanced_rag_service.remove_document_from_index(document_id)
        
        # Delete file
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Delete from database
        db.delete(document)
        db.commit()
        
        return {
            "message": "Document deleted successfully",
            "document_id": document_id,
            "chunks_removed": removed_chunks
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))