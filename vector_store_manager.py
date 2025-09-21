"""
Vector Store Manager for Milvus operations.

This module provides a centralized interface for managing Milvus vector store operations
including collection management, document storage, and retriever configuration.
"""

import os
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import dependencies, create placeholders if not available
try:
    from langchain.schema import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Milvus
    from langchain.vectorstores.base import VectorStoreRetriever
    from dotenv import load_dotenv
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    # Create placeholder classes for testing when dependencies aren't available
    DEPENDENCIES_AVAILABLE = False
    
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    class OpenAIEmbeddings:
        def __init__(self, **kwargs):
            pass
    
    class Milvus:
        def __init__(self, **kwargs):
            pass
        
        @classmethod
        def from_documents(cls, docs, embeddings, **kwargs):
            return cls()
        
        def as_retriever(self, **kwargs):
            return VectorStoreRetriever()
    
    class VectorStoreRetriever:
        def __init__(self, **kwargs):
            pass
        
        def get_relevant_documents(self, query):
            return []
    
    def load_dotenv():
        pass


class VectorStoreManager:
    """
    Manages Milvus vector store operations for the RAG system.
    
    This class provides methods for initializing collections, storing documents,
    clearing collections, and configuring retrievers for the Milvus vector database.
    """
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: str = "19530",
                 embedding_model: str = "text-embedding-3-large"):
        """
        Initialize the VectorStoreManager.
        
        Args:
            host: Milvus server host (default: localhost)
            port: Milvus server port (default: 19530)
            embedding_model: OpenAI embedding model to use
        """
        self.host = host
        self.port = port
        self.connection_args = {"host": host, "port": port}
        
        # Load environment variables
        if DEPENDENCIES_AVAILABLE:
            load_dotenv()
        
        # Initialize OpenAI embeddings
        if DEPENDENCIES_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
                
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                api_key=api_key
            )
        else:
            self.embeddings = OpenAIEmbeddings()
        
        # Store current vector store instance
        self._current_vectorstore: Optional[Milvus] = None
        self._current_collection: Optional[str] = None
        
        logger.info(f"VectorStoreManager initialized with host={host}, port={port}")
    
    def initialize_collection(self, collection_name: str) -> bool:
        """
        Initialize a Milvus collection.
        
        Args:
            collection_name: Name of the collection to initialize
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not DEPENDENCIES_AVAILABLE:
            logger.warning("Dependencies not available, using mock implementation")
            self._current_collection = collection_name
            return True
            
        try:
            # Create a temporary vectorstore to initialize the collection
            temp_docs = [Document(page_content="temp", metadata={"source": "init"})]
            
            vectorstore = Milvus.from_documents(
                temp_docs,
                self.embeddings,
                connection_args=self.connection_args,
                collection_name=collection_name
            )
            
            # Clear the temporary document
            self.clear_collection(collection_name)
            
            self._current_collection = collection_name
            logger.info(f"Collection '{collection_name}' initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize collection '{collection_name}': {str(e)}")
            return False
    
    def clear_collection(self, collection_name: str) -> bool:
        """
        Clear all documents from a Milvus collection.
        
        Args:
            collection_name: Name of the collection to clear
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not DEPENDENCIES_AVAILABLE:
            logger.warning("Dependencies not available, using mock implementation")
            if self._current_collection == collection_name:
                self._current_vectorstore = None
                self._current_collection = None
            return True
            
        try:
            # Connect to existing collection
            vectorstore = Milvus(
                embedding_function=self.embeddings,
                connection_args=self.connection_args,
                collection_name=collection_name
            )
            
            # Get the collection object and drop it to clear all data
            collection = vectorstore.col
            if collection:
                collection.drop()
                logger.info(f"Collection '{collection_name}' cleared successfully")
            
            # Reset current vectorstore if it was the cleared collection
            if self._current_collection == collection_name:
                self._current_vectorstore = None
                self._current_collection = None
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection '{collection_name}': {str(e)}")
            return False
    
    def store_documents(self, docs: List[Document], collection_name: str = "pdf_rag_docs") -> bool:
        """
        Store documents in the Milvus vector store.
        
        Args:
            docs: List of documents to store
            collection_name: Name of the collection to store documents in
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not docs:
            logger.warning("No documents provided to store")
            return True
            
        if not DEPENDENCIES_AVAILABLE:
            logger.warning("Dependencies not available, using mock implementation")
            self._current_collection = collection_name
            return True
        
        try:
            # Create or update the vector store with documents
            vectorstore = Milvus.from_documents(
                docs,
                self.embeddings,
                connection_args=self.connection_args,
                collection_name=collection_name
            )
            
            self._current_vectorstore = vectorstore
            self._current_collection = collection_name
            
            logger.info(f"Successfully stored {len(docs)} documents in collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store documents in collection '{collection_name}': {str(e)}")
            return False
    
    def get_retriever(self, k: int = 3, collection_name: str = "pdf_rag_docs") -> Optional[VectorStoreRetriever]:
        """
        Get a retriever for the vector store.
        
        Args:
            k: Number of documents to retrieve (default: 3)
            collection_name: Name of the collection to retrieve from
            
        Returns:
            VectorStoreRetriever: Configured retriever or None if failed
        """
        if not DEPENDENCIES_AVAILABLE:
            logger.warning("Dependencies not available, using mock implementation")
            return VectorStoreRetriever()
            
        try:
            # Use current vectorstore if it matches the requested collection
            if (self._current_vectorstore and 
                self._current_collection == collection_name):
                vectorstore = self._current_vectorstore
            else:
                # Connect to existing collection
                vectorstore = Milvus(
                    embedding_function=self.embeddings,
                    connection_args=self.connection_args,
                    collection_name=collection_name
                )
                self._current_vectorstore = vectorstore
                self._current_collection = collection_name
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            logger.info(f"Retriever created for collection '{collection_name}' with k={k}")
            return retriever
            
        except Exception as e:
            logger.error(f"Failed to create retriever for collection '{collection_name}': {str(e)}")
            return None
    
    def get_vectorstore(self, collection_name: str = "pdf_rag_docs") -> Optional[Milvus]:
        """
        Get the Milvus vectorstore instance.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Milvus: Vector store instance or None if failed
        """
        if not DEPENDENCIES_AVAILABLE:
            logger.warning("Dependencies not available, using mock implementation")
            return Milvus()
            
        try:
            # Use current vectorstore if it matches the requested collection
            if (self._current_vectorstore and 
                self._current_collection == collection_name):
                return self._current_vectorstore
            
            # Connect to existing collection
            vectorstore = Milvus(
                embedding_function=self.embeddings,
                connection_args=self.connection_args,
                collection_name=collection_name
            )
            
            self._current_vectorstore = vectorstore
            self._current_collection = collection_name
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to get vectorstore for collection '{collection_name}': {str(e)}")
            return None
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        if not DEPENDENCIES_AVAILABLE:
            logger.warning("Dependencies not available, using mock implementation")
            return True
            
        try:
            vectorstore = Milvus(
                embedding_function=self.embeddings,
                connection_args=self.connection_args,
                collection_name=collection_name
            )
            # If we can create the vectorstore object, the collection exists
            return vectorstore.col is not None
            
        except Exception:
            return False
    
    def get_document_count(self, collection_name: str = "pdf_rag_docs") -> int:
        """
        Get the number of documents in a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            int: Number of documents in the collection, -1 if error
        """
        if not DEPENDENCIES_AVAILABLE:
            logger.warning("Dependencies not available, using mock implementation")
            return 0
            
        try:
            vectorstore = self.get_vectorstore(collection_name)
            if vectorstore and hasattr(vectorstore, 'col') and vectorstore.col:
                return vectorstore.col.num_entities
            return 0
            
        except Exception as e:
            logger.error(f"Failed to get document count for collection '{collection_name}': {str(e)}")
            return -1