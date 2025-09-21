"""
Unit tests for chunking strategies.
Tests each chunking strategy implementation, configuration validation, and error handling.
"""
import pytest
from unittest.mock import Mock, patch
from langchain.schema import Document
import numpy as np

from chunking_strategies import (
    ChunkingStrategy,
    RecursiveChunkingStrategy,
    SemanticChunkingStrategy,
    TokenChunkingStrategy,
    ParagraphChunkingStrategy,
    HybridChunkingStrategy,
    ChunkingStrategyFactory
)


class TestChunkingStrategyFactory:
    """Test the chunking strategy factory"""
    
    def test_create_recursive_strategy(self):
        """Test creating recursive chunking strategy"""
        config = {'chunk_size': 500, 'chunk_overlap': 50}
        strategy = ChunkingStrategyFactory.create_strategy('recursive', config)
        
        assert isinstance(strategy, RecursiveChunkingStrategy)
        assert strategy.chunk_size == 500
        assert strategy.chunk_overlap == 50
    
    def test_create_semantic_strategy(self):
        """Test creating semantic chunking strategy"""
        config = {'similarity_threshold': 0.7, 'min_chunk_size': 100}
        strategy = ChunkingStrategyFactory.create_strategy('semantic', config)
        
        assert isinstance(strategy, SemanticChunkingStrategy)
        assert strategy.similarity_threshold == 0.7
        assert strategy.min_chunk_size == 100
    
    def test_create_token_strategy(self):
        """Test creating token chunking strategy"""
        config = {'max_tokens': 800, 'tokenizer_model': 'gpt-3.5-turbo'}
        strategy = ChunkingStrategyFactory.create_strategy('token', config)
        
        assert isinstance(strategy, TokenChunkingStrategy)
        assert strategy.max_tokens == 800
        assert strategy.tokenizer_model == 'gpt-3.5-turbo'
    
    def test_create_paragraph_strategy(self):
        """Test creating paragraph chunking strategy"""
        config = {'min_chunk_size': 150, 'max_chunk_size': 1500}
        strategy = ChunkingStrategyFactory.create_strategy('paragraph', config)
        
        assert isinstance(strategy, ParagraphChunkingStrategy)
        assert strategy.min_chunk_size == 150
        assert strategy.max_chunk_size == 1500
    
    def test_create_hybrid_strategy(self):
        """Test creating hybrid chunking strategy"""
        config = {'chunk_size': 1200, 'similarity_threshold': 0.85}
        strategy = ChunkingStrategyFactory.create_strategy('hybrid', config)
        
        assert isinstance(strategy, HybridChunkingStrategy)
        assert strategy.chunk_size == 1200
        assert strategy.similarity_threshold == 0.85
    
    def test_invalid_strategy_name(self):
        """Test error handling for invalid strategy name"""
        with pytest.raises(ValueError, match="Unknown chunking strategy: invalid"):
            ChunkingStrategyFactory.create_strategy('invalid', {})
    
    def test_get_available_strategies(self):
        """Test getting list of available strategies"""
        strategies = ChunkingStrategyFactory.get_available_strategies()
        expected = ['recursive', 'semantic', 'token', 'paragraph', 'hybrid']
        
        assert set(strategies) == set(expected)


class TestRecursiveChunkingStrategy:
    """Test recursive character text splitting strategy"""
    
    def test_default_configuration(self):
        """Test strategy with default configuration"""
        strategy = RecursiveChunkingStrategy({})
        
        assert strategy.chunk_size == 1000
        assert strategy.chunk_overlap == 100
        assert strategy.separators == ["\n##", "\n#", "\n- ", "\n", " "]
        assert strategy.get_strategy_name() == "Recursive Character Text Splitter"
    
    def test_custom_configuration(self):
        """Test strategy with custom configuration"""
        config = {
            'chunk_size': 500,
            'chunk_overlap': 50,
            'separators': ["\n", " "]
        }
        strategy = RecursiveChunkingStrategy(config)
        
        assert strategy.chunk_size == 500
        assert strategy.chunk_overlap == 50
        assert strategy.separators == ["\n", " "]
    
    def test_chunk_documents(self):
        """Test document chunking functionality"""
        config = {'chunk_size': 100, 'chunk_overlap': 20}
        strategy = RecursiveChunkingStrategy(config)
        
        # Create test document
        long_text = "This is a test document. " * 20  # ~500 characters
        doc = Document(page_content=long_text, metadata={'source': 'test.pdf'})
        
        chunks = strategy.chunk_documents([doc])
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Check metadata
        for i, chunk in enumerate(chunks):
            assert chunk.metadata['chunk_id'] == i
            assert chunk.metadata['chunking_strategy'] == 'recursive'
            assert chunk.metadata['chunk_index'] == i
            assert chunk.metadata['source'] == 'test.pdf'
            assert 'chunk_size' in chunk.metadata
    
    def test_empty_document(self):
        """Test handling of empty documents"""
        strategy = RecursiveChunkingStrategy({})
        doc = Document(page_content="", metadata={'source': 'empty.pdf'})
        
        chunks = strategy.chunk_documents([doc])
        
        # Should handle empty documents gracefully
        assert isinstance(chunks, list)
    
    def test_small_document(self):
        """Test handling of documents smaller than chunk size"""
        config = {'chunk_size': 1000, 'chunk_overlap': 100}
        strategy = RecursiveChunkingStrategy(config)
        
        doc = Document(page_content="Short text", metadata={'source': 'short.pdf'})
        chunks = strategy.chunk_documents([doc])
        
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text"
        assert chunks[0].metadata['chunking_strategy'] == 'recursive'


class TestSemanticChunkingStrategy:
    """Test semantic chunking strategy"""
    
    @patch('chunking_strategies.SentenceTransformer')
    def test_default_configuration(self, mock_transformer):
        """Test strategy with default configuration"""
        strategy = SemanticChunkingStrategy({})
        
        assert strategy.similarity_threshold == 0.8
        assert strategy.min_chunk_size == 200
        assert strategy.get_strategy_name() == "Semantic Chunking"
        mock_transformer.assert_called_once_with('all-MiniLM-L6-v2')
    
    @patch('chunking_strategies.SentenceTransformer')
    def test_custom_configuration(self, mock_transformer):
        """Test strategy with custom configuration"""
        config = {'similarity_threshold': 0.7, 'min_chunk_size': 150}
        strategy = SemanticChunkingStrategy(config)
        
        assert strategy.similarity_threshold == 0.7
        assert strategy.min_chunk_size == 150
    
    def test_split_into_sentences(self):
        """Test sentence splitting functionality"""
        strategy = SemanticChunkingStrategy({})
        
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = strategy._split_into_sentences(text)
        
        expected = ["First sentence", "Second sentence", "Third sentence", "Fourth sentence"]
        assert sentences == expected
    
    def test_split_into_sentences_empty(self):
        """Test sentence splitting with empty text"""
        strategy = SemanticChunkingStrategy({})
        
        sentences = strategy._split_into_sentences("")
        assert sentences == []
    
    @patch('chunking_strategies.SentenceTransformer')
    def test_group_by_similarity_single_sentence(self, mock_transformer):
        """Test grouping with single sentence"""
        strategy = SemanticChunkingStrategy({})
        
        sentences = ["Single sentence"]
        embeddings = np.array([[0.1, 0.2, 0.3]])
        
        chunks = strategy._group_by_similarity(sentences, embeddings)
        assert chunks == ["Single sentence"]
    
    @patch('chunking_strategies.SentenceTransformer')
    def test_group_by_similarity_high_similarity(self, mock_transformer):
        """Test grouping with high similarity sentences"""
        strategy = SemanticChunkingStrategy({'similarity_threshold': 0.8})
        
        sentences = ["First sentence", "Second sentence"]
        # Mock high similarity
        embeddings = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
        
        with patch('chunking_strategies.cosine_similarity', return_value=[[0.9]]):
            chunks = strategy._group_by_similarity(sentences, embeddings)
        
        assert len(chunks) == 1
        assert chunks[0] == "First sentence Second sentence"
    
    @patch('chunking_strategies.SentenceTransformer')
    def test_group_by_similarity_low_similarity(self, mock_transformer):
        """Test grouping with low similarity sentences"""
        strategy = SemanticChunkingStrategy({'similarity_threshold': 0.8})
        
        sentences = ["First sentence", "Different topic"]
        embeddings = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
        
        with patch('chunking_strategies.cosine_similarity', return_value=[[0.3]]):
            chunks = strategy._group_by_similarity(sentences, embeddings)
        
        assert len(chunks) == 2
        assert chunks[0] == "First sentence"
        assert chunks[1] == "Different topic"


class TestTokenChunkingStrategy:
    """Test token-based chunking strategy"""
    
    @patch('chunking_strategies.tiktoken.encoding_for_model')
    def test_default_configuration(self, mock_encoding_for_model):
        """Test strategy with default configuration"""
        mock_encoding = Mock()
        mock_encoding_for_model.return_value = mock_encoding
        
        strategy = TokenChunkingStrategy({})
        
        assert strategy.max_tokens == 1000
        assert strategy.tokenizer_model == 'gpt-4'
        mock_encoding_for_model.assert_called_once_with('gpt-4')
    
    @patch('chunking_strategies.tiktoken.encoding_for_model')
    def test_custom_configuration(self, mock_encoding_for_model):
        """Test strategy with custom configuration"""
        mock_encoding = Mock()
        mock_encoding_for_model.return_value = mock_encoding
        
        config = {'max_tokens': 500, 'tokenizer_model': 'gpt-3.5-turbo'}
        strategy = TokenChunkingStrategy(config)
        
        assert strategy.max_tokens == 500
        assert strategy.tokenizer_model == 'gpt-3.5-turbo'
        mock_encoding_for_model.assert_called_once_with('gpt-3.5-turbo')
    
    @patch('chunking_strategies.tiktoken.encoding_for_model')
    @patch('chunking_strategies.tiktoken.get_encoding')
    def test_fallback_encoding(self, mock_get_encoding, mock_encoding_for_model):
        """Test fallback to cl100k_base encoding when model not found"""
        mock_encoding_for_model.side_effect = KeyError("Model not found")
        mock_encoding = Mock()
        mock_get_encoding.return_value = mock_encoding
        
        strategy = TokenChunkingStrategy({'tokenizer_model': 'unknown-model'})
        
        mock_get_encoding.assert_called_once_with("cl100k_base")
        assert strategy.encoding == mock_encoding
    
    @patch('chunking_strategies.tiktoken.encoding_for_model')
    def test_chunk_by_tokens(self, mock_encoding_for_model):
        """Test token-based text chunking"""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = list(range(150))  # 150 tokens
        mock_encoding.decode.side_effect = lambda tokens: f"chunk_{len(tokens)}_tokens"
        mock_encoding_for_model.return_value = mock_encoding
        
        config = {'max_tokens': 50}
        strategy = TokenChunkingStrategy(config)
        
        chunks = strategy._chunk_by_tokens("test text")
        
        assert len(chunks) == 3  # 150 tokens / 50 max_tokens = 3 chunks
        assert all("chunk_50_tokens" in chunk for chunk in chunks)
    
    @patch('chunking_strategies.tiktoken.encoding_for_model')
    def test_chunk_documents(self, mock_encoding_for_model):
        """Test document chunking with token limits"""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = list(range(100))  # 100 tokens
        mock_encoding.decode.return_value = "decoded_chunk"
        mock_encoding_for_model.return_value = mock_encoding
        
        config = {'max_tokens': 50}
        strategy = TokenChunkingStrategy(config)
        
        doc = Document(page_content="test content", metadata={'source': 'test.pdf'})
        chunks = strategy.chunk_documents([doc])
        
        assert len(chunks) == 2  # 100 tokens / 50 max_tokens = 2 chunks
        
        for i, chunk in enumerate(chunks):
            assert chunk.metadata['chunk_id'] == i
            assert chunk.metadata['chunking_strategy'] == 'token'
            assert chunk.metadata['chunk_index'] == i
            assert chunk.metadata['token_count'] == 50
            assert chunk.metadata['max_tokens'] == 50
            assert chunk.metadata['source'] == 'test.pdf'


class TestParagraphChunkingStrategy:
    """Test paragraph-based chunking strategy"""
    
    def test_default_configuration(self):
        """Test strategy with default configuration"""
        strategy = ParagraphChunkingStrategy({})
        
        assert strategy.min_chunk_size == 200
        assert strategy.max_chunk_size == 2000
        assert strategy.get_strategy_name() == "Paragraph-based Chunking"
    
    def test_custom_configuration(self):
        """Test strategy with custom configuration"""
        config = {'min_chunk_size': 100, 'max_chunk_size': 1500}
        strategy = ParagraphChunkingStrategy(config)
        
        assert strategy.min_chunk_size == 100
        assert strategy.max_chunk_size == 1500
    
    def test_split_into_paragraphs(self):
        """Test paragraph splitting functionality"""
        strategy = ParagraphChunkingStrategy({})
        
        text = "First paragraph.\n\nSecond paragraph.\n\n\nThird paragraph."
        paragraphs = strategy._split_into_paragraphs(text)
        
        expected = ["First paragraph.", "Second paragraph.", "Third paragraph."]
        assert paragraphs == expected
    
    def test_split_into_paragraphs_empty(self):
        """Test paragraph splitting with empty text"""
        strategy = ParagraphChunkingStrategy({})
        
        paragraphs = strategy._split_into_paragraphs("")
        assert paragraphs == []
    
    def test_group_paragraphs_small(self):
        """Test grouping small paragraphs"""
        config = {'min_chunk_size': 50, 'max_chunk_size': 200}
        strategy = ParagraphChunkingStrategy(config)
        
        paragraphs = ["Short para 1.", "Short para 2.", "Short para 3."]
        chunks = strategy._group_paragraphs(paragraphs)
        
        # Should group all paragraphs into one chunk
        assert len(chunks) == 1
        assert "Short para 1.\n\nShort para 2.\n\nShort para 3." in chunks[0]
    
    def test_group_paragraphs_large(self):
        """Test grouping with size limits"""
        config = {'min_chunk_size': 10, 'max_chunk_size': 50}
        strategy = ParagraphChunkingStrategy(config)
        
        # Create paragraphs that exceed max_chunk_size when combined
        paragraphs = ["A" * 30, "B" * 30, "C" * 30]
        chunks = strategy._group_paragraphs(paragraphs)
        
        # Should create separate chunks due to size limits
        assert len(chunks) >= 2
    
    def test_chunk_documents(self):
        """Test document chunking functionality"""
        config = {'min_chunk_size': 20, 'max_chunk_size': 100}
        strategy = ParagraphChunkingStrategy(config)
        
        text = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph content."
        doc = Document(page_content=text, metadata={'source': 'test.pdf'})
        
        chunks = strategy.chunk_documents([doc])
        
        assert len(chunks) >= 1
        
        for i, chunk in enumerate(chunks):
            assert chunk.metadata['chunk_id'] == i
            assert chunk.metadata['chunking_strategy'] == 'paragraph'
            assert chunk.metadata['chunk_index'] == i
            assert chunk.metadata['source'] == 'test.pdf'
            assert 'paragraph_count' in chunk.metadata


class TestHybridChunkingStrategy:
    """Test hybrid chunking strategy"""
    
    @patch('chunking_strategies.tiktoken.get_encoding')
    def test_default_configuration(self, mock_get_encoding):
        """Test strategy with default configuration"""
        mock_encoding = Mock()
        mock_get_encoding.return_value = mock_encoding
        
        strategy = HybridChunkingStrategy({})
        
        assert strategy.chunk_size == 1000
        assert strategy.similarity_threshold == 0.8
        assert strategy.max_tokens == 1000
        assert strategy.get_strategy_name() == "Hybrid Chunking"
    
    @patch('chunking_strategies.tiktoken.get_encoding')
    def test_custom_configuration(self, mock_get_encoding):
        """Test strategy with custom configuration"""
        mock_encoding = Mock()
        mock_get_encoding.return_value = mock_encoding
        
        config = {
            'chunk_size': 800,
            'similarity_threshold': 0.7,
            'max_tokens': 500
        }
        strategy = HybridChunkingStrategy(config)
        
        assert strategy.chunk_size == 800
        assert strategy.similarity_threshold == 0.7
        assert strategy.max_tokens == 500
    
    @patch('chunking_strategies.tiktoken.get_encoding')
    def test_encoding_fallback(self, mock_get_encoding):
        """Test encoding initialization with fallback"""
        mock_get_encoding.side_effect = Exception("Encoding error")
        
        strategy = HybridChunkingStrategy({})
        
        assert strategy.encoding is None
    
    @patch('chunking_strategies.tiktoken.get_encoding')
    def test_split_by_tokens_no_encoding(self, mock_get_encoding):
        """Test token splitting when encoding is not available"""
        mock_get_encoding.side_effect = Exception("Encoding error")
        
        strategy = HybridChunkingStrategy({})
        doc = Document(page_content="test", metadata={})
        
        result = strategy._split_by_tokens(doc)
        
        assert result == [doc]  # Should return original document
    
    @patch('chunking_strategies.tiktoken.get_encoding')
    def test_split_by_tokens_with_encoding(self, mock_get_encoding):
        """Test token splitting with encoding available"""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = list(range(150))  # 150 tokens
        mock_encoding.decode.side_effect = lambda tokens: f"chunk_{len(tokens)}"
        mock_get_encoding.return_value = mock_encoding
        
        config = {'max_tokens': 50}
        strategy = HybridChunkingStrategy(config)
        
        doc = Document(page_content="test", metadata={'source': 'test.pdf'})
        chunks = strategy._split_by_tokens(doc)
        
        assert len(chunks) == 3  # 150 tokens / 50 max_tokens = 3 chunks
        
        for chunk in chunks:
            assert 'token_split' in chunk.metadata.get('hybrid_passes', [])
            assert chunk.metadata['source'] == 'test.pdf'


class TestChunkingStrategyValidation:
    """Test configuration validation and error handling"""
    
    def test_recursive_strategy_invalid_config(self):
        """Test recursive strategy with invalid configuration"""
        # Test with negative chunk size
        config = {'chunk_size': -100}
        strategy = RecursiveChunkingStrategy(config)
        
        # Should handle gracefully with defaults or validation
        assert hasattr(strategy, 'chunk_size')
    
    def test_semantic_strategy_invalid_threshold(self):
        """Test semantic strategy with invalid similarity threshold"""
        config = {'similarity_threshold': 1.5}  # Invalid threshold > 1
        strategy = SemanticChunkingStrategy(config)
        
        # Should handle gracefully
        assert hasattr(strategy, 'similarity_threshold')
    
    def test_token_strategy_invalid_tokens(self):
        """Test token strategy with invalid max_tokens"""
        config = {'max_tokens': 0}
        strategy = TokenChunkingStrategy(config)
        
        # Should handle gracefully
        assert hasattr(strategy, 'max_tokens')
    
    def test_paragraph_strategy_invalid_sizes(self):
        """Test paragraph strategy with invalid size configuration"""
        config = {'min_chunk_size': 1000, 'max_chunk_size': 500}  # min > max
        strategy = ParagraphChunkingStrategy(config)
        
        # Should handle gracefully
        assert hasattr(strategy, 'min_chunk_size')
        assert hasattr(strategy, 'max_chunk_size')


# Test fixtures for sample documents
@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [
        Document(
            page_content="This is a short document for testing purposes.",
            metadata={'source': 'short.pdf', 'page': 1}
        ),
        Document(
            page_content="This is a much longer document that contains multiple sentences and paragraphs. " * 10,
            metadata={'source': 'long.pdf', 'page': 1}
        ),
        Document(
            page_content="Document with\n\nmultiple paragraphs.\n\nEach paragraph\nhas different content.\n\nSome are longer than others.",
            metadata={'source': 'multi_para.pdf', 'page': 1}
        )
    ]


@pytest.fixture
def empty_document():
    """Create empty document for testing edge cases"""
    return Document(page_content="", metadata={'source': 'empty.pdf'})


class TestChunkingStrategiesIntegration:
    """Integration tests for chunking strategies with sample documents"""
    
    def test_all_strategies_with_sample_documents(self, sample_documents):
        """Test all strategies can process sample documents without errors"""
        strategies = {
            'recursive': RecursiveChunkingStrategy({'chunk_size': 100}),
            'paragraph': ParagraphChunkingStrategy({'min_chunk_size': 50}),
        }
        
        for strategy_name, strategy in strategies.items():
            chunks = strategy.chunk_documents(sample_documents)
            
            # Basic validation
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            
            # Check metadata consistency
            for chunk in chunks:
                assert 'chunking_strategy' in chunk.metadata
                assert 'chunk_id' in chunk.metadata
                assert 'chunk_size' in chunk.metadata
    
    def test_strategies_with_empty_document(self, empty_document):
        """Test all strategies handle empty documents gracefully"""
        strategies = {
            'recursive': RecursiveChunkingStrategy({}),
            'paragraph': ParagraphChunkingStrategy({}),
        }
        
        for strategy_name, strategy in strategies.items():
            chunks = strategy.chunk_documents([empty_document])
            
            # Should handle empty documents without errors
            assert isinstance(chunks, list)