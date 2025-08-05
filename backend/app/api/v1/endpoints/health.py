from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.vector_store import vector_store
from app.core.config import settings

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.VERSION
    }

@router.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """Detailed health check including all services"""
    
    try:
        # Check database
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check vector store
    try:
        vector_stats = vector_store.get_stats()
        vector_status = "healthy"
    except Exception as e:
        vector_status = f"unhealthy: {str(e)}"
        vector_stats = {}
    
    # Check Gemini API (simple test)
    try:
        from app.services.llm_service import gemini_service
        # This is a lightweight test - just check if service is initialized
        gemini_status = "healthy" if gemini_service else "unhealthy"
    except Exception as e:
        gemini_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy" if all(
            status == "healthy" for status in [db_status, vector_status, gemini_status]
        ) else "degraded",
        "services": {
            "database": db_status,
            "vector_store": {
                "status": vector_status,
                "stats": vector_stats
            },
            "gemini_api": gemini_status
        },
        "config": {
            "app_name": settings.APP_NAME,
            "version": settings.VERSION,
            "gemini_model": settings.GEMINI_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL
        }
    }