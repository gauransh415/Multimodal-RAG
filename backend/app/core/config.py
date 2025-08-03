from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "ChikuCare RAG API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Database
    DATABASE_URL: str = "sqlite:///./chikucare.db"
    
    # Gemini API
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-1.5-pro"
    GEMINI_VISION_MODEL: str = "gemini-1.5-pro"
    
    # Vector Store
    VECTOR_STORE_PATH: str = "./data/vector_store"
    EMBEDDING_MODEL: str = "text-embedding-004"
    VECTOR_DIMENSION: int = 768
    
    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [
        ".pdf", ".txt", ".docx", ".md",
        ".jpg", ".jpeg", ".png", ".gif", ".webp"
    ]
    UPLOAD_PATH: str = "./data/uploads"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()