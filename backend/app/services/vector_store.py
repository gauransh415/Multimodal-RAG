import numpy as np
import pickle
import os
import uuid
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings

@dataclass
class VectorDocument:
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    document_id: Optional[int] = None
    chunk_index: Optional[int] = None

class VectorStore:
    def __init__(self, store_path: str = None):
        self.store_path = store_path or settings.VECTOR_STORE_PATH
        self.documents: Dict[str, VectorDocument] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        
        # Create directory if it doesn't exist
        os.makedirs(self.store_path, exist_ok=True)
        
        # Load existing store
        self.load_store()
    
    def add_document(
        self, 
        content: str, 
        embedding: List[float], 
        metadata: Dict[str, Any] = None,
        document_id: Optional[int] = None,
        chunk_index: Optional[int] = None
    ) -> str:
        """Add a document to the vector store"""
        doc_id = str(uuid.uuid4())
        
        vector_doc = VectorDocument(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            document_id=document_id,
            chunk_index=chunk_index
        )
        
        self.documents[doc_id] = vector_doc
        self._rebuild_matrix()
        self.save_store()
        
        return doc_id
    
    def add_documents(self, documents: List[Tuple[str, List[float], Dict[str, Any]]]) -> List[str]:
        """Add multiple documents to the vector store"""
        doc_ids = []
        for content, embedding, metadata in documents:
            doc_id = str(uuid.uuid4())
            vector_doc = VectorDocument(
                id=doc_id,
                content=content,
                embedding=embedding,
                metadata=metadata or {}
            )
            self.documents[doc_id] = vector_doc
            doc_ids.append(doc_id)
        
        self._rebuild_matrix()
        self.save_store()
        return doc_ids
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[VectorDocument, float]]:
        """Search for similar documents"""
        if not self.documents or self.embeddings_matrix is None:
            return []
        
        query_vector = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_vector, self.embeddings_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                doc_id = self.doc_ids[idx]
                document = self.documents[doc_id]
                results.append((document, float(similarity)))
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by ID"""
        return self.documents.get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._rebuild_matrix()
            self.save_store()
            return True
        return False
    
    def delete_by_document_id(self, document_id: int) -> int:
        """Delete all chunks belonging to a document"""
        to_delete = [
            doc_id for doc_id, doc in self.documents.items() 
            if doc.document_id == document_id
        ]
        
        for doc_id in to_delete:
            del self.documents[doc_id]
        
        if to_delete:
            self._rebuild_matrix()
            self.save_store()
        
        return len(to_delete)
    
    def _rebuild_matrix(self):
        """Rebuild the embeddings matrix and doc_ids list"""
        if not self.documents:
            self.embeddings_matrix = None
            self.doc_ids = []
            return
        
        self.doc_ids = list(self.documents.keys())
        embeddings = [self.documents[doc_id].embedding for doc_id in self.doc_ids]
        self.embeddings_matrix = np.array(embeddings)
    
    def save_store(self):
        """Save the vector store to disk"""
        store_file = os.path.join(self.store_path, "vector_store.pkl")
        
        # Prepare data for serialization
        store_data = {
            'documents': {
                doc_id: {
                    'id': doc.id,
                    'content': doc.content,
                    'embedding': doc.embedding,
                    'metadata': doc.metadata,
                    'document_id': doc.document_id,
                    'chunk_index': doc.chunk_index
                }
                for doc_id, doc in self.documents.items()
            }
        }
        
        with open(store_file, 'wb') as f:
            pickle.dump(store_data, f)
    
    def load_store(self):
        """Load the vector store from disk"""
        store_file = os.path.join(self.store_path, "vector_store.pkl")
        
        if os.path.exists(store_file):
            try:
                with open(store_file, 'rb') as f:
                    store_data = pickle.load(f)
                
                # Reconstruct documents
                self.documents = {}
                for doc_id, doc_data in store_data.get('documents', {}).items():
                    self.documents[doc_id] = VectorDocument(**doc_data)
                
                self._rebuild_matrix()
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.documents = {}
                self.embeddings_matrix = None
                self.doc_ids = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': len(list(self.documents.values())[0].embedding) if self.documents else 0,
            'store_path': self.store_path
        }
    
    def clear(self):
        """Clear all documents from the store"""
        self.documents = {}
        self.embeddings_matrix = None
        self.doc_ids = []
        self.save_store()

# Global instance
vector_store = VectorStore()