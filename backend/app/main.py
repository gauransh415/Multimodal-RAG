from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
import uvicorn
from app.core.config import settings
from app.api.v1.api import api_router
from app.core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initialization")    
    # Initialize database
    await init_db()
    
    # Create upload directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs("./data/knowledge_base/documents", exist_ok=True)
    os.makedirs("./data/knowledge_base/images", exist_ok=True)
    
    print("Backend initialized successfully!")
    yield
    
    # Shutdown
    print("Shutting down ChikuCare...")


app = FastAPI(
    title="ChikuCare RAG API",
    description="Multimodal RAG system for cat care and health queries",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "ChikuCare RAG System API", "status": "active"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )