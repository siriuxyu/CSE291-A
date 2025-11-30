"""
Vector Storage Implementation

Storage team can choose: ChromaDB, Weaviate, Pinecone, or PostgreSQL with pgvector.
"""

import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

from agent.interfaces import (
    StorageDocument,
    SearchResult,
    StorageStats,
    StorageType,
    StorageError,
    ConnectionError,
)
from .embedding_service import create_embedding_service

logger = logging.getLogger(__name__)

class VectorStorageBackend:
    """
    Vector Database Storage Backend
    Implementation using ChromaDB
    """

    def __init__(self, collection_name: str = "memory_context", **config):
        """
        Initialize vector storage configuration
        """
        if chromadb is None:
            raise ImportError("chromadb not installed. Please run `pip install chromadb`")

        self.collection_name = collection_name
        self.config = config
        self.client = None
        self.collection = None
        self.embedding_service = None
        self.persist_path = config.get("persist_path", "./chroma_db_data")

    # ========================================================================
    # PUBLIC INTERFACE - Called by Memory team
    # ========================================================================

    async def initialize(self) -> None:
        """Initialize storage backend connection"""
        try:
            logger.info(f"Initializing VectorStorage at {self.persist_path}...")
            
            # 1. Init Embedding Service
            self.embedding_service = create_embedding_service(
                provider=self.config.get("embedding_provider", "openai"),
                api_key=self.config.get("openai_api_key")
            )

            # 2. Init ChromaDB
            self.client = chromadb.PersistentClient(path=self.persist_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("VectorStorage initialized successfully.")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize VectorStorage: {e}")

    async def store_document(
        self, document: StorageDocument, generate_embedding: bool = True
    ) -> str:
        """Store a document in vector database"""
        try:
            # Ensure ID exists
            doc_id = document.id if document.id else str(uuid.uuid4())
            
            # Handle Embedding
            embedding = document.embedding
            if generate_embedding and not embedding:
                embedding = await self.embedding_service.embed_text(document.content)
            
            if not embedding:
                raise StorageError("Document missing embedding")

            # Prepare data for Chroma (flatten metadata)
            final_metadata = self._prepare_metadata(document)
            
            self.collection.upsert(
                ids=[doc_id],
                documents=[document.content],
                embeddings=[embedding],
                metadatas=[final_metadata]
            )
            return doc_id
        except Exception as e:
            raise StorageError(f"Store document failed: {e}")

    async def store_documents_batch(
        self, documents: List[StorageDocument], generate_embeddings: bool = True
    ) -> List[str]:
        """Store multiple documents in batch"""
        if not documents:
            return []
        
        try:
            ids = []
            texts = []
            metadatas = []
            embeddings = []
            
            # Identify documents needing embedding
            docs_needing_embedding = []
            indices_needing_embedding = []

            for i, doc in enumerate(documents):
                doc_id = doc.id if doc.id else str(uuid.uuid4())
                ids.append(doc_id)
                texts.append(doc.content)
                metadatas.append(self._prepare_metadata(doc))
                
                if doc.embedding:
                    embeddings.append(doc.embedding)
                else:
                    embeddings.append(None) # Placeholder
                    if generate_embeddings:
                        docs_needing_embedding.append(doc.content)
                        indices_needing_embedding.append(i)

            # Batch generate embeddings
            if docs_needing_embedding:
                new_embeddings = await self.embedding_service.embed_texts_batch(docs_needing_embedding)
                for i, idx in enumerate(indices_needing_embedding):
                    embeddings[idx] = new_embeddings[i]
            
            # Check for missing embeddings
            if any(e is None for e in embeddings):
                 raise StorageError("Some documents failed to have embeddings.")

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            return ids
        except Exception as e:
            raise StorageError(f"Batch store failed: {e}")

    async def search_similar(
        self,
        query_text: str,
        user_id: str,
        document_types: Optional[List[StorageType]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents using vector similarity"""
        try:
            # 1. Generate Query embedding
            query_embedding = await self.embedding_service.embed_text(query_text)
            
            # 2. Call search by embedding
            return await self.search_by_embedding(
                embedding=query_embedding,
                user_id=user_id,
                document_types=document_types,
                top_k=top_k,
                similarity_threshold=similarity_threshold
                # filter_metadata TODO: Parameter not present in search_by_embedding signature, skipped for now
            )
        except Exception as e:
            raise StorageError(f"Search similar failed: {e}")

    async def search_by_embedding(
        self,
        embedding: List[float],
        user_id: str,
        document_types: Optional[List[StorageType]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Search using pre-computed embedding"""
        
        # Build filter criteria
        where_clause = {"user_id": user_id}
        
        if document_types:
            types_vals = [t.value for t in document_types]
            if len(types_vals) == 1:
                where_clause["document_type"] = types_vals[0]
            else:
                where_clause["document_type"] = {"$in": types_vals}

        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
            
            search_results = []
            if results["ids"]:
                for i in range(len(results["ids"][0])):
                    distance = results["distances"][0][i]
                    score = 1 - distance # Assuming Cosine Distance
                    
                    if score < similarity_threshold:
                        continue
                        
                    doc = self._convert_to_storage_document(
                        res_id=results["ids"][0][i],
                        res_doc=results["documents"][0][i],
                        res_meta=results["metadatas"][0][i],
                        res_emb=results["embeddings"][0][i]
                    )
                    search_results.append(SearchResult(
                        document=doc,
                        similarity_score=score,
                        rank=i+1
                    ))
            return search_results
            
        except Exception as e:
            raise StorageError(f"Search by embedding failed: {e}")

    async def get_document(
        self, document_id: str, user_id: str
    ) -> Optional[StorageDocument]:
        """Retrieve a specific document by ID"""
        try:
            # Chroma get method
            results = self.collection.get(
                ids=[document_id],
                where={"user_id": user_id},
                include=["documents", "metadatas", "embeddings"]
            )
            
            if results["ids"]:
                return self._convert_to_storage_document(
                    res_id=results["ids"][0],
                    res_doc=results["documents"][0],
                    res_meta=results["metadatas"][0],
                    res_emb=results["embeddings"][0]
                )
            return None
        except Exception as e:
            raise StorageError(f"Get document failed: {e}")

    async def get_documents_by_session(
        self,
        session_id: str,
        user_id: str,
        document_types: Optional[List[StorageType]] = None,
    ) -> List[StorageDocument]:
        """Retrieve all documents for a session"""
        where_clause = {"session_id": session_id, "user_id": user_id}
        
        if document_types:
            types_vals = [t.value for t in document_types]
            if len(types_vals) == 1:
                where_clause["document_type"] = types_vals[0]
            else:
                where_clause["document_type"] = {"$in": types_vals}

        try:
            results = self.collection.get(
                where=where_clause,
                include=["documents", "metadatas", "embeddings"]
            )
            
            docs = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    docs.append(self._convert_to_storage_document(
                        res_id=results["ids"][i],
                        res_doc=results["documents"][i],
                        res_meta=results["metadatas"][i],
                        res_emb=results["embeddings"][i]
                    ))
            return docs
        except Exception as e:
            raise StorageError(f"Get session documents failed: {e}")

    async def get_user_contexts(
        self,
        user_id: str,
        document_types: Optional[List[StorageType]] = None,
        limit: int = 100,
    ) -> List[StorageDocument]:
        """Retrieve long-term contexts for a user"""
        # Contexts usually refer to Summary or Long Term Memory
        where_clause = {"user_id": user_id}
        
        if document_types:
            types_vals = [t.value for t in document_types]
            if len(types_vals) == 1:
                where_clause["document_type"] = types_vals[0]
            else:
                where_clause["document_type"] = {"$in": types_vals}

        try:
            results = self.collection.get(
                where=where_clause,
                limit=limit,
                include=["documents", "metadatas", "embeddings"]
            )
            
            docs = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    docs.append(self._convert_to_storage_document(
                        res_id=results["ids"][i],
                        res_doc=results["documents"][i],
                        res_meta=results["metadatas"][i],
                        res_emb=results["embeddings"][i]
                    ))
            return docs
        except Exception as e:
            raise StorageError(f"Get user contexts failed: {e}")

    async def update_document(
        self, document_id: str, updates: Dict[str, Any], user_id: str
    ) -> bool:
        """Update document metadata or content"""
        # Chroma update is simpler, but here we need complex logic for content vs metadata
        # For simplicity, we Fetch then Upsert
        try:
            existing = await self.get_document(document_id, user_id)
            if not existing:
                return False
            
            # Apply updates
            new_content = updates.get("content", existing.content)
            new_metadata = existing.metadata.copy()
            new_metadata.update(updates.get("metadata", {}))
            
            # Re-embed if content changed
            embedding = existing.embedding
            if "content" in updates:
                 embedding = await self.embedding_service.embed_text(new_content)

            # Re-construct Metadata Flattening
            # Note: Update StorageDocument fields first
            existing.content = new_content
            existing.metadata = new_metadata
            # If session_id is in updates, handle it here...
            
            final_metadata = self._prepare_metadata(existing)

            self.collection.update(
                ids=[document_id],
                documents=[new_content],
                embeddings=[embedding] if embedding else None,
                metadatas=[final_metadata]
            )
            return True
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False

    async def delete_document(self, document_id: str, user_id: str) -> bool:
        """Delete a document"""
        try:
            # Add user_id for security
            self.collection.delete(
                ids=[document_id],
                where={"user_id": user_id}
            )
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    async def delete_session_documents(
        self, session_id: str, user_id: str, preserve_summaries: bool = True
    ) -> int:
        """Delete all documents for a session"""
        where_clause = {"session_id": session_id, "user_id": user_id}
        
        if preserve_summaries:
            # Do not delete SUMMARY type
            where_clause["document_type"] = {"$ne": StorageType.SESSION_SUMMARY.value}

        try:
            # Chroma delete does not return count, so this is best-effort
            self.collection.delete(where=where_clause)
            return 0 # Cannot determine count
        except Exception as e:
            raise StorageError(f"Delete session failed: {e}")

    async def get_storage_stats(self, user_id: Optional[str] = None) -> StorageStats:
        """Get storage usage statistics"""
        # Chroma count() is global. Filtering by user_id requires get(where=...) then len()
        # This might be slow for large datasets.
        try:
            if user_id:
                res = self.collection.get(where={"user_id": user_id}, include=[])
                count = len(res["ids"])
            else:
                count = self.collection.count()
                
            return StorageStats(
                total_documents=count,
                documents_by_type={}, # Aggregation is hard in Chroma
                total_size_bytes=0,
                oldest_document=datetime.now(),
                newest_document=datetime.now()
            )
        except Exception as e:
            raise StorageError(f"Get stats failed: {e}")

    async def close(self) -> None:
        """Close storage connections"""
        # Chroma PersistentClient manages file handles automatically, usually no need to close explicitly
        pass

    # ========================================================================
    # INTERNAL HELPERS - Storage team only
    # ========================================================================

    def _get_embedding_service(self):
        """Get embedding service instance"""
        return self.embedding_service

    def _prepare_metadata(self, doc: StorageDocument) -> Dict[str, Any]:
        """Flatten StorageDocument into Chroma metadata"""
        meta = doc.metadata.copy() if doc.metadata else {}
        # Inject core fields
        meta["user_id"] = doc.user_id
        meta["document_type"] = doc.document_type.value # Store Enum Value (str)
        meta["created_at"] = doc.created_at.isoformat()
        meta["updated_at"] = doc.updated_at.isoformat()
        if doc.session_id:
            meta["session_id"] = doc.session_id
        return meta

    def _convert_to_storage_document(
        self, res_id, res_doc, res_meta, res_emb
    ) -> StorageDocument:
        """Convert raw vector DB result to StorageDocument"""
        
        # Extract and remove core fields
        user_id = res_meta.pop("user_id", "")
        doc_type_val = res_meta.pop("document_type", StorageType.SHORT_TERM_MESSAGE.value)
        session_id = res_meta.pop("session_id", None)
        created_at_str = res_meta.pop("created_at", datetime.now().isoformat())
        updated_at_str = res_meta.pop("updated_at", datetime.now().isoformat())
        
        # Restore Enum
        try:
            doc_type = StorageType(doc_type_val)
        except ValueError:
            doc_type = StorageType.SHORT_TERM_MESSAGE

        return StorageDocument(
            id=res_id,
            user_id=user_id,
            session_id=session_id,
            document_type=doc_type,
            content=res_doc,
            embedding=res_emb,
            metadata=res_meta, # Remaining are original Custom Metadata
            created_at=datetime.fromisoformat(created_at_str),
            updated_at=datetime.fromisoformat(updated_at_str)
        )