from typing import List, Dict, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass
import json
from PIL import Image
import io

from app.services.llm_service import gemini_service
from app.services.vector_store import vector_store, VectorDocument
from app.services.document_processor import DocumentProcessor
from app.core.config import settings
from app.models.document import Document
from sqlalchemy.orm import Session

@dataclass
class EnhancedRAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query: str
    reasoning: str
    document_types: List[str]

class EnhancedRAGService:
    def __init__(self):
        self.top_k = settings.TOP_K_RETRIEVAL
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.document_processor = DocumentProcessor()
    
    async def multimodal_query(
        self, 
        query: str, 
        images: List[bytes] = None,
        session_context: List[Dict] = None,
        top_k: Optional[int] = None
    ) -> EnhancedRAGResponse:
        """Enhanced multimodal RAG query with context awareness"""
        
        top_k = top_k or self.top_k
        
        try:
            # Step 1: Process images if provided
            image_descriptions = []
            if images:
                for i, image_data in enumerate(images):
                    description = await gemini_service.analyze_image(
                        image_data, 
                        f"Analyze this image in the context of the query: '{query}'. "
                        "Provide detailed description focusing on relevant elements."
                    )
                    image_descriptions.append(f"Image {i+1}: {description}")
            
            # Step 2: Enhance query with image context
            enhanced_query = query
            if image_descriptions:
                enhanced_query = f"{query}\n\nImage Context:\n" + "\n".join(image_descriptions)
            
            # Step 3: Generate query embedding
            query_embedding = await gemini_service.generate_query_embedding(enhanced_query)
            
            # Step 4: Retrieve relevant documents
            retrieved_docs = await self._intelligent_retrieval(query_embedding, enhanced_query, top_k)
            
            # Step 5: Prepare context with ranking
            context = self._prepare_ranked_context(retrieved_docs, enhanced_query)
            
            # Step 6: Generate comprehensive response
            system_prompt = self._create_system_prompt(session_context)
            
            if images:
                # Multimodal response
                full_content = [
                    f"{system_prompt}\n\nContext from knowledge base:\n{context}\n\nUser query: {query}"
                ]
                full_content.extend([Image.open(io.BytesIO(img_data)) for img_data in images])
                
                response = await gemini_service.multimodal_query(
                    text_prompt="Based on the provided context and images, provide a comprehensive answer.",
                    images=images,
                    context=context
                )
            else:
                # Text-only response
                response = await gemini_service.generate_text_response(
                    prompt=f"{system_prompt}\n\nContext: {context}\n\nQuery: {enhanced_query}\n\nProvide a comprehensive answer:",
                    temperature=0.7
                )
            
            # Step 7: Generate reasoning explanation
            reasoning = await self._generate_reasoning(enhanced_query, retrieved_docs, response)
            
            # Step 8: Calculate confidence and prepare response
            confidence = self._calculate_enhanced_confidence(retrieved_docs, enhanced_query)
            sources = self._prepare_enhanced_sources(retrieved_docs)
            document_types = list(set([doc.metadata.get('type', 'unknown') for doc in retrieved_docs]))
            
            return EnhancedRAGResponse(
                answer=response,
                sources=sources,
                confidence=confidence,
                query=query,
                reasoning=reasoning,
                document_types=document_types
            )
            
        except Exception as e:
            raise Exception(f"Error in enhanced RAG query: {str(e)}")
    
    async def _intelligent_retrieval(
        self, 
        query_embedding: List[float], 
        enhanced_query: str,
        top_k: int
    ) -> List[VectorDocument]:
        """Intelligent retrieval with query expansion and re-ranking"""
        
        # Initial retrieval
        initial_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more results for re-ranking
            threshold=self.similarity_threshold * 0.8  # Lower initial threshold
        )
        
        if not initial_results:
            return []
        
        # Re-rank results based on query relevance
        reranked_results = await self._rerank_results(initial_results, enhanced_query)
        
        return [doc for doc, score in reranked_results[:top_k]]
    
    async def _rerank_results(
        self, 
        initial_results: List[Tuple[VectorDocument, float]], 
        query: str
    ) -> List[Tuple[VectorDocument, float]]:
        """Re-rank results using LLM-based relevance scoring"""
        
        reranking_tasks = []
        for doc, similarity in initial_results:
            task = self._score_relevance(doc.content, query)
            reranking_tasks.append((doc, similarity, task))
        
        # Execute relevance scoring
        scored_results = []
        for doc, similarity, task in reranking_tasks:
            try:
                relevance_score = await task
                # Combine similarity and relevance scores
                final_score = (similarity * 0.6) + (relevance_score * 0.4)
                scored_results.append((doc, final_score))
            except:
                # Fallback to similarity score
                scored_results.append((doc, similarity))
        
        # Sort by final score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results
    
    async def _score_relevance(self, content: str, query: str) -> float:
        """Score content relevance to query using LLM"""
        
        scoring_prompt = f"""
        Rate the relevance of the following content to the query on a scale of 0.0 to 1.0.
        Only respond with a single number.

        Query: {query}

        Content: {content[:500]}...

        Relevance score (0.0-1.0):
        """
        
        try:
            response = await gemini_service.generate_text_response(
                scoring_prompt, 
                temperature=0.1,
                max_tokens=10
            )
            score = float(response.strip())
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except:
            return 0.5  # Default score if parsing fails
    
    def _prepare_ranked_context(self, docs: List[VectorDocument], query: str) -> str:
        """Prepare context with intelligent ranking and summarization"""
        
        if not docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Add document type information
            doc_type = doc.metadata.get('type', 'document')
            filename = doc.metadata.get('original_filename', 'Unknown')
            
            context_part = f"""
Source {i} ({doc_type} - {filename}):
{doc.content}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_system_prompt(self, session_context: List[Dict] = None) -> str:
        """Create dynamic system prompt based on context"""
        
        base_prompt = """You are an intelligent assistant with access to a knowledge base. 
Your role is to provide accurate, helpful, and comprehensive answers based on the provided context.

Guidelines:
1. Use the provided context to answer questions accurately
2. If the context doesn't contain enough information, say so clearly
3. Cite your sources when possible
4. For multimodal queries, consider both text and image information
5. Provide step-by-step reasoning when helpful
6. Be concise but thorough"""

        if session_context:
            conversation_context = "\n".join([
                f"User: {msg.get('user', '')}\nAssistant: {msg.get('assistant', '')}" 
                for msg in session_context[-3:]  # Last 3 exchanges
            ])
            base_prompt += f"\n\nRecent conversation context:\n{conversation_context}"
        
        return base_prompt
    
    async def _generate_reasoning(
        self, 
        query: str, 
        retrieved_docs: List[VectorDocument], 
        response: str
    ) -> str:
        """Generate explanation of reasoning process"""
        
        doc_info = [
            f"- {doc.metadata.get('original_filename', 'Unknown')} ({doc.metadata.get('type', 'document')})"
            for doc in retrieved_docs[:3]
        ]
        
        reasoning_prompt = f"""
        Explain briefly how you arrived at your answer for the query: "{query}"
        
        You used these sources:
        {chr(10).join(doc_info)}
        
        Your answer was: {response[:200]}...
        
        Provide a brief explanation of your reasoning process:
        """
        
        try:
            reasoning = await gemini_service.generate_text_response(
                reasoning_prompt,
                temperature=0.3,
                max_tokens=200
            )
            return reasoning
        except Exception as e:
            return f"Retrieved information from {len(retrieved_docs)} sources to provide a comprehensive answer."
    
    def _calculate_enhanced_confidence(self, docs: List[VectorDocument], query: str) -> float:
        """Calculate enhanced confidence score"""
        
        if not docs:
            return 0.0
        
        # Base confidence from similarity scores
        avg_similarity = sum(doc.metadata.get('similarity', 0.5) for doc in docs) / len(docs)
        
        # Boost confidence based on document diversity
        doc_types = set(doc.metadata.get('type', 'unknown') for doc in docs)
        diversity_boost = min(len(doc_types) * 0.1, 0.3)
        
        # Consider number of relevant documents
        relevance_factor = min(len(docs) / self.top_k, 1.0) * 0.2
        
        final_confidence = min(avg_similarity + diversity_boost + relevance_factor, 1.0)
        return final_confidence
    
    def _prepare_enhanced_sources(self, docs: List[VectorDocument]) -> List[Dict[str, Any]]:
        """Prepare enhanced source information"""
        
        sources = []
        for doc in docs:
            source = {
                "content_preview": doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
                "filename": doc.metadata.get('original_filename', 'Unknown'),
                "type": doc.metadata.get('type', 'document'),
                "document_id": doc.document_id,
                "chunk_index": doc.chunk_index,
                "relevance_score": doc.metadata.get('similarity', 0.0)
            }
            sources.append(source)
        
        return sources
    
    async def process_and_index_document(
        self, 
        file_path: str, 
        filename: str, 
        db: Session,
        user_id: Optional[int] = None
    ) -> Document:
        """Process and index a new document"""
        return await self.document_processor.process_document(
            file_path, filename, db, user_id
        )

# Enhanced global instance
enhanced_rag_service = EnhancedRAGService()