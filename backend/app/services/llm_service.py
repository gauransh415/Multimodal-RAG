import google.generativeai as genai
from PIL import Image
import io
import base64
from typing import List, Dict, Any, Optional, Union
import json
import asyncio
from functools import wraps

from app.core.config import settings

# Configure Gemini
genai.configure(api_key=settings.GEMINI_API_KEY)

def async_wrap(func):
    @wraps(func)
    async def async_func(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    return async_func

class GeminiService:
    def __init__(self):
        self.text_model = genai.GenerativeModel(settings.GEMINI_MODEL)
        self.vision_model = genai.GenerativeModel(settings.GEMINI_VISION_MODEL)
        
    @async_wrap
    def generate_text_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate text response using Gemini"""
        try:
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
            
            response = self.text_model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            return response.text
        except Exception as e:
            raise Exception(f"Error generating text response: {str(e)}")
    
    @async_wrap
    def analyze_image(
        self, 
        image_data: bytes, 
        prompt: str = "Describe this image in detail."
    ) -> str:
        """Analyze image using Gemini Vision"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            response = self.vision_model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            raise Exception(f"Error analyzing image: {str(e)}")
    
    @async_wrap
    def multimodal_query(
        self, 
        text_prompt: str, 
        images: List[bytes] = None,
        context: Optional[str] = None
    ) -> str:
        """Handle multimodal queries with text and images"""
        try:
            content = []
            
            # Add context if provided
            if context:
                content.append(f"Context: {context}\n\n")
            
            content.append(text_prompt)
            
            # Add images if provided
            if images:
                for image_data in images:
                    image = Image.open(io.BytesIO(image_data))
                    content.append(image)
            
            response = self.vision_model.generate_content(content)
            return response.text
        except Exception as e:
            raise Exception(f"Error in multimodal query: {str(e)}")
    
    @async_wrap
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text using Gemini"""
        try:
            result = genai.embed_content(
                model=f"models/{settings.EMBEDDING_MODEL}",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    @async_wrap
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embeddings for query using Gemini"""
        try:
            result = genai.embed_content(
                model=f"models/{settings.EMBEDDING_MODEL}",
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            raise Exception(f"Error generating query embedding: {str(e)}")

# Global instance
gemini_service = GeminiService()