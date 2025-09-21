"""
Test fixtures for chunking strategy tests.
Provides sample documents and configurations for testing.
"""
import pytest
from langchain.schema import Document
from typing import List, Dict, Any


@pytest.fixture
def sample_short_document():
    """Create a short document for testing"""
    return Document(
        page_content="This is a short document for testing purposes. It contains only a few sentences.",
        metadata={'source': 'short.pdf', 'page': 1}
    )


@pytest.fixture
def sample_long_document():
    """Create a long document for testing chunking"""
    content = """This is the first paragraph of a longer document. It contains multiple sentences and provides substantial content for testing chunking strategies. The paragraph discusses various aspects of document processing and text analysis.

This is the second paragraph. It continues the discussion with additional information about text processing techniques. The content here is designed to test paragraph-based chunking and semantic similarity detection.

The third paragraph introduces new concepts and topics. This helps test the semantic chunking strategy's ability to detect topic changes and create appropriate boundaries. The content shifts to discuss different aspects of the subject matter.

Finally, the fourth paragraph concludes the document. It summarizes the key points and provides closure to the discussion. This paragraph helps test the completeness of chunking strategies."""
    
    return Document(
        page_content=content,
        metadata={'source': 'long.pdf', 'page': 1}
    )


@pytest.fixture
def sample_technical_document():
    """Create a technical document with code and structured content"""
    content = """# Introduction to RAG Systems

RAG (Retrieval-Augmented Generation) systems combine information retrieval with language generation. These systems are particularly useful for question-answering tasks.

## Architecture Components

### Vector Database
- Stores document embeddings
- Enables semantic search
- Common implementations: Milvus, Pinecone, Weaviate

### Text Splitter
The text splitter breaks documents into chunks:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
```

### Embedding Model
Converts text to vector representations for similarity search.

## Implementation Details

When implementing a RAG system, consider:
1. Chunk size optimization
2. Overlap configuration
3. Retrieval strategy
4. Generation model selection

The system performance depends heavily on these configuration choices."""
    
    return Document(
        page_content=content,
        metadata={'source': 'technical.pdf', 'page': 1}
    )


@pytest.fixture
def sample_empty_document():
    """Create an empty document for edge case testing"""
    return Document(
        page_content="",
        metadata={'source': 'empty.pdf', 'page': 1}
    )


@pytest.fixture
def sample_single_sentence_document():
    """Create a document with only one sentence"""
    return Document(
        page_content="This document contains only a single sentence.",
        metadata={'source': 'single.pdf', 'page': 1}
    )


@pytest.fixture
def sample_documents_list(sample_short_document, sample_long_document, sample_technical_document):
    """Create a list of sample documents for batch testing"""
    return [sample_short_document, sample_long_document, sample_technical_document]


@pytest.fixture
def recursive_config_small():
    """Configuration for small recursive chunks"""
    return {
        'chunk_size': 200,
        'chunk_overlap': 50,
        'separators': ["\n##", "\n#", "\n- ", "\n", " "]
    }


@pytest.fixture
def recursive_config_large():
    """Configuration for large recursive chunks"""
    return {
        'chunk_size': 1000,
        'chunk_overlap': 100,
        'separators': ["\n##", "\n#", "\n- ", "\n", " "]
    }


@pytest.fixture
def semantic_config_strict():
    """Configuration for strict semantic chunking"""
    return {
        'similarity_threshold': 0.9,
        'min_chunk_size': 100
    }


@pytest.fixture
def semantic_config_loose():
    """Configuration for loose semantic chunking"""
    return {
        'similarity_threshold': 0.6,
        'min_chunk_size': 50
    }


@pytest.fixture
def token_config_small():
    """Configuration for small token chunks"""
    return {
        'max_tokens': 100,
        'tokenizer_model': 'gpt-3.5-turbo'
    }


@pytest.fixture
def token_config_large():
    """Configuration for large token chunks"""
    return {
        'max_tokens': 500,
        'tokenizer_model': 'gpt-4'
    }


@pytest.fixture
def paragraph_config_small():
    """Configuration for small paragraph chunks"""
    return {
        'min_chunk_size': 50,
        'max_chunk_size': 300
    }


@pytest.fixture
def paragraph_config_large():
    """Configuration for large paragraph chunks"""
    return {
        'min_chunk_size': 200,
        'max_chunk_size': 1500
    }


@pytest.fixture
def hybrid_config_balanced():
    """Configuration for balanced hybrid chunking"""
    return {
        'chunk_size': 800,
        'similarity_threshold': 0.8,
        'max_tokens': 600
    }


@pytest.fixture
def hybrid_config_aggressive():
    """Configuration for aggressive hybrid chunking"""
    return {
        'chunk_size': 400,
        'similarity_threshold': 0.9,
        'max_tokens': 300
    }


@pytest.fixture
def invalid_configs():
    """Dictionary of invalid configurations for testing error handling"""
    return {
        'recursive_negative_size': {
            'chunk_size': -100,
            'chunk_overlap': 50
        },
        'recursive_overlap_larger_than_size': {
            'chunk_size': 100,
            'chunk_overlap': 200
        },
        'semantic_invalid_threshold': {
            'similarity_threshold': 1.5,  # > 1.0
            'min_chunk_size': 100
        },
        'semantic_negative_threshold': {
            'similarity_threshold': -0.1,  # < 0.0
            'min_chunk_size': 100
        },
        'token_zero_tokens': {
            'max_tokens': 0,
            'tokenizer_model': 'gpt-4'
        },
        'token_negative_tokens': {
            'max_tokens': -100,
            'tokenizer_model': 'gpt-4'
        },
        'paragraph_min_greater_than_max': {
            'min_chunk_size': 1000,
            'max_chunk_size': 500
        },
        'paragraph_zero_sizes': {
            'min_chunk_size': 0,
            'max_chunk_size': 0
        }
    }


@pytest.fixture
def expected_chunk_counts():
    """Expected chunk counts for different strategies and documents"""
    return {
        'short_document': {
            'recursive_small': 1,  # Short document should create 1 chunk
            'recursive_large': 1,
            'paragraph_small': 1,
            'paragraph_large': 1
        },
        'long_document': {
            'recursive_small': 4,  # Approximate expected chunks
            'recursive_large': 2,
            'paragraph_small': 4,  # 4 paragraphs
            'paragraph_large': 1   # All paragraphs fit in one chunk
        }
    }


class DocumentGenerator:
    """Utility class for generating test documents"""
    
    @staticmethod
    def create_document_with_length(target_length: int, source: str = "generated.pdf") -> Document:
        """Create a document with approximately the target length"""
        base_text = "This is a test sentence for document generation. "
        repeat_count = max(1, target_length // len(base_text))
        content = base_text * repeat_count
        
        # Trim to exact length if needed
        if len(content) > target_length:
            content = content[:target_length].rsplit(' ', 1)[0]  # Don't cut words
        
        return Document(
            page_content=content,
            metadata={'source': source, 'generated_length': len(content)}
        )
    
    @staticmethod
    def create_document_with_paragraphs(paragraph_count: int, source: str = "paragraphs.pdf") -> Document:
        """Create a document with a specific number of paragraphs"""
        paragraphs = []
        for i in range(paragraph_count):
            paragraph = f"This is paragraph number {i + 1}. It contains multiple sentences to provide adequate content for testing. The paragraph discusses topic {i + 1} in detail."
            paragraphs.append(paragraph)
        
        content = "\n\n".join(paragraphs)
        return Document(
            page_content=content,
            metadata={'source': source, 'paragraph_count': paragraph_count}
        )
    
    @staticmethod
    def create_document_with_tokens(target_tokens: int, source: str = "tokens.pdf") -> Document:
        """Create a document with approximately the target token count"""
        # Rough estimate: 1 token ≈ 4 characters for English text
        target_length = target_tokens * 4
        return DocumentGenerator.create_document_with_length(target_length, source)


@pytest.fixture
def document_generator():
    """Provide the DocumentGenerator utility"""
    return DocumentGenerator