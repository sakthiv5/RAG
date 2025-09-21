"""
Chunking strategies for document processing.
Implements various text chunking approaches for RAG systems.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents according to the strategy.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents with metadata
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of the chunking strategy"""
        pass

class RecursiveChunkingStrategy(ChunkingStrategy):
    """Recursive character text splitting strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 100)
        self.separators = config.get('separators', ["\n##", "\n#", "\n- ", "\n", " "])
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents using recursive character splitting"""
        chunks = self.splitter.split_documents(documents)
        
        # Add chunking metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'chunking_strategy': 'recursive',
                'chunk_size': len(chunk.page_content),
                'chunk_index': i
            })
        
        return chunks
    
    def get_strategy_name(self) -> str:
        return "Recursive Character Text Splitter"

class SemanticChunkingStrategy(ChunkingStrategy):
    """Semantic chunking based on sentence similarity"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        self.min_chunk_size = config.get('min_chunk_size', 200)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents using semantic similarity"""
        all_chunks = []
        
        for doc in documents:
            # Split into sentences
            sentences = self._split_into_sentences(doc.page_content)
            if not sentences:
                continue
            
            # Get embeddings for sentences
            embeddings = self.model.encode(sentences)
            
            # Group sentences by similarity
            chunks = self._group_by_similarity(sentences, embeddings)
            
            # Create Document objects
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text) >= self.min_chunk_size:
                    chunk_doc = Document(
                        page_content=chunk_text,
                        metadata={
                            **doc.metadata,
                            'chunk_id': len(all_chunks),
                            'chunking_strategy': 'semantic',
                            'chunk_size': len(chunk_text),
                            'chunk_index': i,
                            'similarity_threshold': self.similarity_threshold
                        }
                    )
                    all_chunks.append(chunk_doc)
        
        return all_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be improved with spaCy or NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _group_by_similarity(self, sentences: List[str], embeddings: np.ndarray) -> List[str]:
        """Group sentences by semantic similarity"""
        if len(sentences) <= 1:
            return sentences
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0:1]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            chunk_mean = np.mean(current_embedding, axis=0).reshape(1, -1)
            similarity = cosine_similarity(chunk_mean, embeddings[i:i+1])[0][0]
            
            if similarity >= self.similarity_threshold:
                # Add to current chunk
                current_chunk.append(sentences[i])
                current_embedding = np.vstack([current_embedding, embeddings[i:i+1]])
            else:
                # Start new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i:i+1]
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def get_strategy_name(self) -> str:
        return "Semantic Chunking"

class TokenChunkingStrategy(ChunkingStrategy):
    """Token-based chunking using tiktoken"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_tokens = config.get('max_tokens', 1000)
        self.tokenizer_model = config.get('tokenizer_model', 'gpt-4')
        
        try:
            self.encoding = tiktoken.encoding_for_model(self.tokenizer_model)
        except KeyError:
            # Fallback to cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents based on token count"""
        all_chunks = []
        
        for doc in documents:
            chunks = self._chunk_by_tokens(doc.page_content)
            
            for i, chunk_text in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        'chunk_id': len(all_chunks),
                        'chunking_strategy': 'token',
                        'chunk_size': len(chunk_text),
                        'chunk_index': i,
                        'token_count': len(self.encoding.encode(chunk_text)),
                        'max_tokens': self.max_tokens
                    }
                )
                all_chunks.append(chunk_doc)
        
        return all_chunks
    
    def _chunk_by_tokens(self, text: str) -> List[str]:
        """Split text into chunks based on token count"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def get_strategy_name(self) -> str:
        return "Token-based Chunking"

class ParagraphChunkingStrategy(ChunkingStrategy):
    """Paragraph-based chunking strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_chunk_size = config.get('min_chunk_size', 200)
        self.max_chunk_size = config.get('max_chunk_size', 2000)
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents by paragraphs"""
        all_chunks = []
        
        for doc in documents:
            paragraphs = self._split_into_paragraphs(doc.page_content)
            chunks = self._group_paragraphs(paragraphs)
            
            for i, chunk_text in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        'chunk_id': len(all_chunks),
                        'chunking_strategy': 'paragraph',
                        'chunk_size': len(chunk_text),
                        'chunk_index': i,
                        'paragraph_count': chunk_text.count('\n\n') + 1
                    }
                )
                all_chunks.append(chunk_doc)
        
        return all_chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _group_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Group paragraphs into appropriately sized chunks"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            para_size = len(paragraph)
            
            # If adding this paragraph would exceed max size, start new chunk
            if current_size + para_size > self.max_chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                current_chunk = [paragraph]
                current_size = para_size
            else:
                current_chunk.append(paragraph)
                current_size += para_size
        
        # Add the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def get_strategy_name(self) -> str:
        return "Paragraph-based Chunking"

class HybridChunkingStrategy(ChunkingStrategy):
    """Hybrid chunking combining multiple strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.chunk_size = config.get('chunk_size', 1000)
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        self.max_tokens = config.get('max_tokens', 1000)
        
        # Initialize sub-strategies
        self.recursive_strategy = RecursiveChunkingStrategy({
            'chunk_size': self.chunk_size,
            'chunk_overlap': 50,
            'separators': ["\n##", "\n#", "\n- ", "\n", " "]
        })
        
        self.semantic_strategy = SemanticChunkingStrategy({
            'similarity_threshold': self.similarity_threshold,
            'min_chunk_size': 100
        })
        
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Apply hybrid chunking strategy"""
        # First pass: recursive chunking
        initial_chunks = self.recursive_strategy.chunk_documents(documents)
        
        # Second pass: semantic refinement for large chunks
        refined_chunks = []
        
        for chunk in initial_chunks:
            if len(chunk.page_content) > self.chunk_size * 1.5:
                # Apply semantic chunking to large chunks
                semantic_chunks = self.semantic_strategy.chunk_documents([chunk])
                for sem_chunk in semantic_chunks:
                    sem_chunk.metadata.update({
                        'chunking_strategy': 'hybrid',
                        'hybrid_passes': ['recursive', 'semantic']
                    })
                refined_chunks.extend(semantic_chunks)
            else:
                chunk.metadata.update({
                    'chunking_strategy': 'hybrid',
                    'hybrid_passes': ['recursive']
                })
                refined_chunks.append(chunk)
        
        # Third pass: token limit enforcement
        final_chunks = []
        for chunk in refined_chunks:
            if self.encoding and len(self.encoding.encode(chunk.page_content)) > self.max_tokens:
                # Split by tokens if too large
                token_chunks = self._split_by_tokens(chunk)
                final_chunks.extend(token_chunks)
            else:
                final_chunks.append(chunk)
        
        # Update chunk IDs
        for i, chunk in enumerate(final_chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_index'] = i
        
        return final_chunks
    
    def _split_by_tokens(self, document: Document) -> List[Document]:
        """Split a document by token limits"""
        if not self.encoding:
            return [document]
        
        tokens = self.encoding.encode(document.page_content)
        chunks = []
        
        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **document.metadata,
                    'chunk_size': len(chunk_text),
                    'token_count': len(chunk_tokens),
                    'hybrid_passes': document.metadata.get('hybrid_passes', []) + ['token_split']
                }
            )
            chunks.append(chunk_doc)
        
        return chunks
    
    def get_strategy_name(self) -> str:
        return "Hybrid Chunking"

class ChunkingStrategyFactory:
    """Factory for creating chunking strategies"""
    
    _strategies = {
        'recursive': RecursiveChunkingStrategy,
        'semantic': SemanticChunkingStrategy,
        'token': TokenChunkingStrategy,
        'paragraph': ParagraphChunkingStrategy,
        'hybrid': HybridChunkingStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, config: Dict[str, Any]) -> ChunkingStrategy:
        """Create a chunking strategy instance"""
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown chunking strategy: {strategy_name}")
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class(config)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy names"""
        return list(cls._strategies.keys())