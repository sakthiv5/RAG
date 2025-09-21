"""
RAG Service - Main orchestrator for query processing and configuration management
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv

from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from chunking_strategies import ChunkingStrategyFactory
from error_handler import error_handler, handle_errors, safe_execute, ErrorRecovery

# Load environment variables
load_dotenv()


@dataclass
class QueryResponse:
    """Response structure for RAG queries"""
    answer: str
    sources: List[str]
    processing_time: float
    chunk_count: int
    error: Optional[str] = None


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies"""
    strategy: str
    chunk_size: int = 1000
    chunk_overlap: int = 100
    separators: Optional[List[str]] = None
    similarity_threshold: Optional[float] = None
    max_tokens: Optional[int] = None
    tokenizer_model: Optional[str] = None


class RAGService:
    """Main orchestrator for RAG operations"""
    
    def __init__(self):
        """Initialize RAG service with default configuration"""
        error_handler.logger.info("Initializing RAG service...")
        
        # Validate API key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize components with error handling
        try:
            self.document_processor = DocumentProcessor()
            error_handler.logger.info("Document processor initialized")
        except Exception as e:
            error_handler.handle_startup_error(e, "document_processor")
            raise
        
        try:
            self.vector_store_manager = VectorStoreManager()
            error_handler.logger.info("Vector store manager initialized")
        except Exception as e:
            error_handler.handle_startup_error(e, "vector_store")
            raise
        
        # Initialize LLM with retry logic
        try:
            def init_llm():
                return ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=self.openai_api_key,
                    temperature=0.1,
                    timeout=30,
                    max_retries=2
                )
            
            self.llm = ErrorRecovery.retry_with_backoff(init_llm, max_retries=3)
            error_handler.logger.info("OpenAI LLM initialized successfully")
        except Exception as e:
            error_handler.handle_startup_error(e, "openai")
            raise
        
        # Initialize embeddings with retry logic
        try:
            def init_embeddings():
                return OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    api_key=self.openai_api_key,
                    timeout=30,
                    max_retries=2
                )
            
            self.embeddings = ErrorRecovery.retry_with_backoff(init_embeddings, max_retries=3)
            error_handler.logger.info("OpenAI embeddings initialized successfully")
        except Exception as e:
            error_handler.handle_startup_error(e, "openai")
            raise
        
        # Default chunking configuration
        self.chunking_config = ChunkingConfig(
            strategy="recursive_character",
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n##", "\n#", "\n- ", "\n", " "]
        )
        
        # Load saved configuration if exists
        safe_execute(
            self._load_chunking_config,
            context="load_chunking_config"
        )
        
        error_handler.logger.info("RAG service initialization completed")
    
    @handle_errors("query_processing")
    def query(self, question: str) -> QueryResponse:
        """
        Process a user question and return an answer with source attribution
        
        Args:
            question: User's question string
            
        Returns:
            QueryResponse with answer, sources, and metadata
        """
        start_time = datetime.now()
        error_handler.logger.info(f"Processing query: {question[:100]}...")
        
        try:
            # Get vector store and create retriever with error handling
            vectorstore = safe_execute(
                self.vector_store_manager.get_vectorstore,
                "pdf_rag_docs",  # <-- FIX: pass collection name as string
                default_return=None,
                context="get_vectorstore"
            )
            
            if not vectorstore:
                error_handler.logger.warning("Vector store not available for query")
                return QueryResponse(
                    answer="",
                    sources=[],
                    processing_time=0.0,
                    chunk_count=0,
                    error="No documents have been processed yet. Please upload and process documents first."
                )
            
            # Create retriever with error handling
            try:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            except Exception as e:
                error_handler.log_error(e, "create_retriever")
                return QueryResponse(
                    answer="",
                    sources=[],
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    chunk_count=0,
                    error="Error accessing document database. Please try reprocessing documents."
                )
            
            # Create QA chain with error handling
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type="stuff"
                )
            except Exception as e:
                error_handler.log_error(e, "create_qa_chain")
                return QueryResponse(
                    answer="",
                    sources=[],
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    chunk_count=0,
                    error="Error initializing question-answering system. Please try again."
                )
            
            # Process query with retry logic
            def process_query():
                return qa_chain.invoke({"query": question})
            
            result = ErrorRecovery.retry_with_backoff(
                process_query,
                max_retries=2,
                backoff_factor=1.0
            )
            
            # Extract sources safely
            sources = []
            source_docs = result.get("source_documents", [])
            for doc in source_docs:
                source = doc.metadata.get("source", "Unknown")
                if source not in sources:
                    sources.append(source)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            error_handler.logger.info(f"Query processed successfully in {processing_time:.2f}s")
            
            return QueryResponse(
                answer=result.get("result", "No answer generated"),
                sources=sources,
                processing_time=processing_time,
                chunk_count=len(source_docs)
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = error_handler.log_error(e, "query_processing")
            
            return QueryResponse(
                answer="",
                sources=[],
                processing_time=processing_time,
                chunk_count=0,
                error=error_msg
            )
    
    def get_chunking_strategies(self) -> Dict[str, Any]:
        """
        Get available chunking strategies and current configuration
        
        Returns:
            Dictionary with available strategies and current config
        """
        strategies = {
            "recursive_character": {
                "name": "Recursive Character Text Splitter",
                "description": "Default recursive character text splitter with configurable separators",
                "config_options": ["chunk_size", "chunk_overlap", "separators"]
            },
            "semantic": {
                "name": "Semantic Chunking",
                "description": "Embedding-based semantic chunking using sentence similarity",
                "config_options": ["chunk_size", "similarity_threshold"]
            },
            "token_based": {
                "name": "Token-based Chunking",
                "description": "Token-aware chunking respecting model token limits",
                "config_options": ["max_tokens", "tokenizer_model", "chunk_overlap"]
            },
            "paragraph": {
                "name": "Paragraph-based Chunking",
                "description": "Split on paragraph boundaries with size constraints",
                "config_options": ["chunk_size", "chunk_overlap"]
            },
            "hybrid": {
                "name": "Hybrid Chunking",
                "description": "Combination of multiple chunking strategies",
                "config_options": ["chunk_size", "chunk_overlap", "similarity_threshold"]
            }
        }
        
        return {
            "strategies": strategies,
            "current_config": asdict(self.chunking_config)
        }
    
    @handle_errors("update_chunking_config")
    def update_chunking_config(self, strategy: str, config: Dict[str, Any]) -> bool:
        """
        Update chunking configuration
        
        Args:
            strategy: Name of the chunking strategy
            config: Configuration parameters
            
        Returns:
            True if successful, False otherwise
        """
        error_handler.logger.info(f"Updating chunking config: strategy={strategy}")
        
        try:
            # Validate strategy
            available_strategies = ["recursive_character", "semantic", "token_based", "paragraph", "hybrid"]
            if strategy not in available_strategies:
                raise ValueError(f"Invalid strategy '{strategy}'. Available strategies: {', '.join(available_strategies)}")
            
            # Store previous config for rollback
            previous_config = ChunkingConfig(**asdict(self.chunking_config))
            
            # Update configuration with validation
            self.chunking_config.strategy = strategy
            
            # Update relevant config parameters with type validation
            try:
                if "chunk_size" in config:
                    chunk_size = int(config["chunk_size"])
                    if not (100 <= chunk_size <= 10000):
                        raise ValueError("chunk_size must be between 100 and 10000")
                    self.chunking_config.chunk_size = chunk_size
                
                if "chunk_overlap" in config:
                    chunk_overlap = int(config["chunk_overlap"])
                    if not (0 <= chunk_overlap <= 1000):
                        raise ValueError("chunk_overlap must be between 0 and 1000")
                    self.chunking_config.chunk_overlap = chunk_overlap
                
                if "separators" in config:
                    separators = config["separators"]
                    if isinstance(separators, str):
                        separators = [s.strip() for s in separators.split(',')]
                    self.chunking_config.separators = separators
                
                if "similarity_threshold" in config:
                    threshold = float(config["similarity_threshold"])
                    if not (0.1 <= threshold <= 1.0):
                        raise ValueError("similarity_threshold must be between 0.1 and 1.0")
                    self.chunking_config.similarity_threshold = threshold
                
                if "max_tokens" in config:
                    max_tokens = int(config["max_tokens"])
                    if not (100 <= max_tokens <= 8000):
                        raise ValueError("max_tokens must be between 100 and 8000")
                    self.chunking_config.max_tokens = max_tokens
                
                if "tokenizer_model" in config:
                    self.chunking_config.tokenizer_model = str(config["tokenizer_model"])
                
            except (ValueError, TypeError) as e:
                # Rollback to previous config
                self.chunking_config = previous_config
                raise ValueError(f"Invalid configuration parameter: {str(e)}")
            
            # Save configuration with error handling
            save_success = safe_execute(
                self._save_chunking_config,
                default_return=False,
                context="save_chunking_config"
            )
            
            if not save_success:
                error_handler.logger.warning("Failed to save chunking config to file")
            
            error_handler.logger.info(f"Chunking configuration updated successfully: {strategy}")
            return True
            
        except Exception as e:
            error_handler.log_error(e, "update_chunking_config")
            return False
    
    @handle_errors("reprocess_documents")
    def reprocess_documents(self) -> bool:
        """
        Reprocess documents with current chunking configuration
        
        Returns:
            True if successful, False otherwise
        """
        error_handler.logger.info("Starting document reprocessing")
        
        try:
            # Clear existing vector store with error handling
            clear_success = safe_execute(
                self.vector_store_manager.clear_collection,
                "pdf_rag_docs", 
                default_return=False,
                context="clear_vector_store"
            )
            
            if not clear_success:
                error_handler.logger.warning("Failed to clear vector store, continuing anyway")
            
            # Process documents with current chunking config
            error_handler.logger.info("Converting PDF documents")
            documents = safe_execute(
                self.document_processor.convert_pdfs,
                default_return=[],
                context="convert_pdfs"
            )
            
            if not documents:
                error_handler.logger.warning("No documents found to process")
                return False
            
            error_handler.logger.info(f"Found {len(documents)} documents to process")
            
            # Chunk documents using current strategy with error handling
            error_handler.logger.info(f"Chunking documents with strategy: {self.chunking_config.strategy}")
            
            def chunk_with_fallback():
                try:
                    return self.document_processor.chunk_documents(
                        documents,
                        self.chunking_config.strategy,
                        asdict(self.chunking_config)
                    )
                except Exception as e:
                    error_handler.logger.warning(f"Chunking failed with {self.chunking_config.strategy}, falling back to recursive")
                    # Fallback to recursive strategy
                    fallback_config = {
                        'strategy': 'recursive_character',
                        'chunk_size': 1000,
                        'chunk_overlap': 100,
                        'separators': ["\n##", "\n#", "\n- ", "\n", " "]
                    }
                    return self.document_processor.chunk_documents(
                        documents,
                        'recursive_character',
                        fallback_config
                    )
            
            chunks = ErrorRecovery.retry_with_backoff(
                chunk_with_fallback,
                max_retries=2,
                backoff_factor=1.0
            )
            
            if not chunks:
                error_handler.logger.error("No chunks created from documents")
                return False
            
            error_handler.logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            
            # Store in vector database with retry logic
            def store_with_retry():
                return self.vector_store_manager.store_documents(chunks, "pdf_rag_docs")
            
            success = ErrorRecovery.retry_with_backoff(
                store_with_retry,
                max_retries=3,
                backoff_factor=2.0
            )
            
            if success:
                error_handler.logger.info(f"Successfully reprocessed {len(documents)} documents into {len(chunks)} chunks")
            else:
                error_handler.logger.error("Failed to store chunks in vector database")
            
            return success
            
        except Exception as e:
            error_handler.log_error(e, "reprocess_documents")
            return False
    
    def _load_chunking_config(self):
        """Load chunking configuration from file"""
        config_file = "config/chunking_config.json"
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    self.chunking_config = ChunkingConfig(**config_data)
                    error_handler.logger.info(f"Loaded chunking config: {self.chunking_config.strategy}")
            else:
                error_handler.logger.info("No existing chunking config found, using defaults")
        except Exception as e:
            error_handler.log_error(e, "load_chunking_config", "Could not load chunking configuration")
    
    def _save_chunking_config(self):
        """Save chunking configuration to file"""
        config_file = "config/chunking_config.json"
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(asdict(self.chunking_config), f, indent=2)
            error_handler.logger.info(f"Saved chunking config to {config_file}")
        except Exception as e:
            error_handler.log_error(e, "save_chunking_config", "Could not save chunking configuration")