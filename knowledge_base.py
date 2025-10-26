"""
Vector Knowledge Base using ChromaDB
Provides RAG (Retrieval Augmented Generation) capabilities
"""
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Optional


class VectorKnowledgeBase:
    """Vector database for storing and retrieving documents"""
    
    def __init__(self, persist_directory: str = "./vector_db"):
        """Initialize ChromaDB client"""
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Error initializing collection: {e}")
            self.collection = None
    
    async def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a document to the knowledge base"""
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        doc_id = str(uuid.uuid4())
        
        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata or {}],
                ids=[doc_id]
            )
            return doc_id
        except Exception as e:
            print(f"Error adding document: {e}")
            raise
    
    async def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add multiple documents to the knowledge base"""
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        
        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas or [{} for _ in texts],
                ids=doc_ids
            )
            return doc_ids
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise
    
    async def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Search for relevant documents"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            # Extract documents from results
            if results and results['documents']:
                return results['documents'][0]
            return []
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    async def search_with_metadata(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search and return documents with metadata"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            # Combine documents with metadata
            documents = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    documents.append({
                        'id': results['ids'][0][i],
                        'document': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results.get('distances') else None
                    })
            
            return documents
            
        except Exception as e:
            print(f"Error searching with metadata: {e}")
            return []
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        if not self.collection:
            return False
        
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        if not self.collection:
            return {"count": 0, "error": "Collection not initialized"}
        
        try:
            return {
                "count": self.collection.count(),
                "name": self.collection.name,
                "metadata": self.collection.metadata
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"count": 0, "error": str(e)}
    
    async def health_check(self) -> bool:
        """Check if knowledge base is operational"""
        try:
            if not self.collection:
                return False
            # Try to get count as health check
            self.collection.count()
            return True
        except:
            return False
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        if self.collection:
            try:
                self.client.delete_collection(name="knowledge_base")
                self.collection = self.client.create_collection(
                    name="knowledge_base",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                print(f"Error clearing collection: {e}")
