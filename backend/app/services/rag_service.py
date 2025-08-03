from typing import List, Dict, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass

from app.services.llm_service import gemini_service
from app.services.vector_store import vector_store, VectorDocument
from app.core.config import settings

@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query: str

@dataclass
class RetrievedDocument:
    content: str
    similarity: float
    metadata: Dict[str, Any]
    document_id: Optional[int] = None
    chunk_index: Optional[int] = None

class RAGService:
    def __init__(self):
        self.top_k = settings.TOP_K_RETRIEVAL
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
    
    async def query(
        self, 
        query: str, 
        images: List[bytes] = None,
        top_k: Optional[int] = None,
        include_sources: bool = True
    ) -> RAGResponse:
        """
        Process a RAG query with optional multimodal input
        """
        top_k = top_k or self.top_k
        
        try:
            # Step 1: Generate query embedding
            query_embedding = await gemini_service.generate_query_embedding(query)
            
            # Step 2: Retrieve relevant documents
            retrieved_docs = await self._retrieve_documents(query_embedding, top_k)
            
            # Step 3: Prepare context from retrieved documents
            context = self._prepare_context(retrieved_docs)
            
            # Step 4: Generate response
            if images:
                # Multimodal query
                response = await gemini_service.multimodal_query(
                    text_prompt=query,
                    images=images,
                    context=context
                )
            else:
                # Text-only query
                response = await gemini_service.generate_text_response(
                    prompt=query,
                    context=context
                )
            
            # Step 5: Calculate confidence based on retrieval scores
            confidence = self._calculate_confidence(retrieved_docs)
            
            # Step 6: Prepare sources
            sources = []
            if include_sources:
                sources = self._prepare_sources(retrieved_docs)
            
            return RAGResponse(
                answer=response,
                sources=sources,
                confidence=confidence,
                query=query
            )
            
        except Exception as e:
            raise Exception(f"Error in RAG query: {str(e)}")
    
    async def _retrieve_documents(
        self, 
        query_embedding: List[float], 
        top_k: int
    ) -> List[RetrievedDocument]:
        """Retrieve relevant documents from vector store"""
        
        # Search in vector store
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=self.similarity_threshold
        )
        
        retrieved_docs = []
        for vector_doc, similarity in results:
            retrieved_doc = RetrievedDocument(
                content=vector_doc.content,
                similarity=similarity,
                metadata=vector_doc.metadata,
                document_id=vector_doc.document_id,
                chunk_index=vector_doc.chunk_index
            )
            retrieved_docs.append(retrieved_doc)
        
        return retrieved_docs
    
    def _prepare_context(self, retrieved_docs: List[RetrievedDocument]) -> str:
        """Prepare context string from retrieved documents"""
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_part = f"Document {i} (Relevance: {doc.similarity:.2f}):\n{doc.content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(self, retrieved_docs: List[RetrievedDocument]) -> float:
        """Calculate confidence score based on retrieval results"""
        if not retrieved_docs:
            return 0.0
        
        # Simple confidence calculation based on top similarity score
        # and number of relevant documents
        top_similarity = retrieved_docs[0].similarity if retrieved_docs else 0.0
        
        # Weight by number of relevant documents found
        num_relevant = len([doc for doc in retrieved_docs if doc.similarity > 0.5])
        relevance_factor = min(num_relevant / self.top_k, 1.0)
        
        confidence = (top_similarity * 0.7) + (relevance_factor * 0.3)
        return min(confidence, 1.0)
    
    def _prepare_sources(self, retrieved_docs: List[RetrievedDocument]) -> List[Dict[str, Any]]:
        """Prepare source information for response"""
        sources = []
        for doc in retrieved_docs:
            source = {
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "similarity": doc.similarity,
                "metadata": doc.metadata,
                "document_id": doc.document_id,
                "chunk_index": doc.chunk_index
            }
            sources.append(source)
        
        return sources
    
    async def add_documents_to_index(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add documents to the vector index
        documents: List of dicts with 'content', 'metadata', 'document_id', 'chunk_index'
        """
        doc_ids = []
        
        for doc in documents:
            content = doc['content']
            metadata = doc.get('metadata', {})
            document_id = doc.get('document_id')
            chunk_index = doc.get('chunk_index')
            
            # Generate embedding
            embedding = await gemini_service.generate_embeddings(content)
            
            # Add to vector store
            doc_id = vector_store.add_document(
                content=content,
                embedding=embedding,
                metadata=metadata,
                document_id=document_id,
                chunk_index=chunk_index
            )
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def remove_document_from_index(self, document_id: int) -> int:
        """Remove all chunks of a document from the index"""
        return vector_store.delete_by_document_id(document_id)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index"""
        return vector_store.get_stats()

# Global instance
rag_service = RAGService()