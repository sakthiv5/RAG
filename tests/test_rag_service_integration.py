"""
Integration tests for RAG service.
Tests complete query processing flow, OpenAI integration, and document reprocessing workflows.
"""
import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from dataclasses import asdict

from rag_service import RAGService, QueryResponse, ChunkingConfig
from langchain.schema import Document


class TestRAGServiceIntegration:
    """Integration tests for RAG service functionality"""
    
    @pytest.fixture
    def mock_openai_api_key(self, monkeypatch):
        """Mock OpenAI API key for testing"""
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies"""
        with patch('rag_service.DocumentProcessor') as mock_doc_processor, \
             patch('rag_service.VectorStoreManager') as mock_vector_manager, \
             patch('rag_service.ChatOpenAI') as mock_llm, \
             patch('rag_service.OpenAIEmbeddings') as mock_embeddings, \
             patch('rag_service.ErrorRecovery'):
            
            # Setup mock instances
            mock_doc_processor_instance = Mock()
            mock_vector_manager_instance = Mock()
            mock_llm_instance = Mock()
            mock_embeddings_instance = Mock()
            
            mock_doc_processor.return_value = mock_doc_processor_instance
            mock_vector_manager.return_value = mock_vector_manager_instance
            mock_llm.return_value = mock_llm_instance
            mock_embeddings.return_value = mock_embeddings_instance
            
            yield {
                'doc_processor': mock_doc_processor_instance,
                'vector_manager': mock_vector_manager_instance,
                'llm': mock_llm_instance,
                'embeddings': mock_embeddings_instance
            }
    
    def test_rag_service_initialization(self, mock_openai_api_key, mock_dependencies):
        """Test RAG service initializes correctly with all components"""
        service = RAGService()
        
        assert service.openai_api_key == "test-api-key"
        assert service.document_processor is not None
        assert service.vector_store_manager is not None
        assert service.llm is not None
        assert service.embeddings is not None
        assert isinstance(service.chunking_config, ChunkingConfig)
    
    def test_query_processing_flow_success(self, mock_openai_api_key, mock_dependencies):
        """Test complete query processing flow with successful response"""
        service = RAGService()
        
        # Mock vector store and retriever
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_dependencies['vector_manager'].get_vectorstore.return_value = mock_vectorstore
        
        # Mock QA chain response
        mock_qa_result = {
            'result': 'This is the answer to your question.',
            'source_documents': [
                Mock(metadata={'source': 'document1.pdf'}),
                Mock(metadata={'source': 'document2.pdf'})
            ]
        }
        
        with patch('rag_service.RetrievalQA') as mock_qa:
            mock_qa_chain = Mock()
            mock_qa_chain.invoke.return_value = mock_qa_result
            mock_qa.from_chain_type.return_value = mock_qa_chain
            
            response = service.query("What is the main topic?")
        
        assert isinstance(response, QueryResponse)
        assert response.answer == 'This is the answer to your question.'
        assert len(response.sources) == 2
        assert 'document1.pdf' in response.sources
        assert 'document2.pdf' in response.sources
        assert response.chunk_count == 2
        assert response.processing_time > 0
        assert response.error is None
    
    def test_query_processing_no_vectorstore(self, mock_openai_api_key, mock_dependencies):
        """Test query processing when no vector store is available"""
        service = RAGService()
        
        # Mock no vector store available
        mock_dependencies['vector_manager'].get_vectorstore.return_value = None
        
        response = service.query("What is the main topic?")
        
        assert isinstance(response, QueryResponse)
        assert response.answer == ""
        assert response.sources == []
        assert response.chunk_count == 0
        assert response.error == "No documents have been processed yet. Please upload and process documents first."
    
    def test_query_processing_openai_error(self, mock_openai_api_key, mock_dependencies):
        """Test query processing with OpenAI API error"""
        service = RAGService()
        
        # Mock vector store setup
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_dependencies['vector_manager'].get_vectorstore.return_value = mock_vectorstore
        
        # Mock QA chain that raises an exception
        with patch('rag_service.RetrievalQA') as mock_qa:
            mock_qa_chain = Mock()
            mock_qa_chain.invoke.side_effect = Exception("OpenAI API error")
            mock_qa.from_chain_type.return_value = mock_qa_chain
            
            response = service.query("What is the main topic?")
        
        assert isinstance(response, QueryResponse)
        assert response.answer == ""
        assert response.sources == []
        assert response.chunk_count == 0
        assert response.error is not None
        assert "error" in response.error.lower()
    
    def test_get_chunking_strategies(self, mock_openai_api_key, mock_dependencies):
        """Test getting available chunking strategies"""
        service = RAGService()
        
        strategies_data = service.get_chunking_strategies()
        
        assert 'strategies' in strategies_data
        assert 'current_config' in strategies_data
        
        # Check all expected strategies are present
        expected_strategies = ['recursive_character', 'semantic', 'token_based', 'paragraph', 'hybrid']
        for strategy in expected_strategies:
            assert strategy in strategies_data['strategies']
        
        # Check current config structure
        current_config = strategies_data['current_config']
        assert 'strategy' in current_config
        assert 'chunk_size' in current_config
        assert 'chunk_overlap' in current_config
    
    def test_update_chunking_config_success(self, mock_openai_api_key, mock_dependencies):
        """Test successful chunking configuration update"""
        service = RAGService()
        
        new_config = {
            'chunk_size': 800,
            'chunk_overlap': 80,
            'separators': ['\n\n', '\n', ' ']
        }
        
        with patch.object(service, '_save_chunking_config'):
            success = service.update_chunking_config('recursive_character', new_config)
        
        assert success is True
        assert service.chunking_config.strategy == 'recursive_character'
        assert service.chunking_config.chunk_size == 800
        assert service.chunking_config.chunk_overlap == 80
        assert service.chunking_config.separators == ['\n\n', '\n', ' ']
    
    def test_update_chunking_config_invalid_strategy(self, mock_openai_api_key, mock_dependencies):
        """Test chunking configuration update with invalid strategy"""
        service = RAGService()
        
        success = service.update_chunking_config('invalid_strategy', {})
        
        assert success is False
    
    def test_update_chunking_config_invalid_parameters(self, mock_openai_api_key, mock_dependencies):
        """Test chunking configuration update with invalid parameters"""
        service = RAGService()
        
        # Test invalid chunk size
        invalid_config = {'chunk_size': -100}
        success = service.update_chunking_config('recursive_character', invalid_config)
        assert success is False
        
        # Test invalid chunk overlap
        invalid_config = {'chunk_overlap': 2000}
        success = service.update_chunking_config('recursive_character', invalid_config)
        assert success is False
        
        # Test invalid similarity threshold
        invalid_config = {'similarity_threshold': 1.5}
        success = service.update_chunking_config('semantic', invalid_config)
        assert success is False
    
    def test_reprocess_documents_success(self, mock_openai_api_key, mock_dependencies):
        """Test successful document reprocessing workflow"""
        service = RAGService()
        
        # Mock document processing
        mock_documents = [
            Document(page_content="Test document 1", metadata={'source': 'test1.pdf'}),
            Document(page_content="Test document 2", metadata={'source': 'test2.pdf'})
        ]
        mock_dependencies['doc_processor'].convert_pdfs.return_value = mock_documents
        
        # Mock chunking
        mock_chunks = [
            Document(page_content="Chunk 1", metadata={'source': 'test1.pdf', 'chunk_id': 0}),
            Document(page_content="Chunk 2", metadata={'source': 'test1.pdf', 'chunk_id': 1}),
            Document(page_content="Chunk 3", metadata={'source': 'test2.pdf', 'chunk_id': 2})
        ]
        mock_dependencies['doc_processor'].chunk_documents.return_value = mock_chunks
        
        # Mock vector store operations
        mock_dependencies['vector_manager'].clear_collection.return_value = True
        mock_dependencies['vector_manager'].store_documents.return_value = True
        
        success = service.reprocess_documents()
        
        assert success is True
        mock_dependencies['vector_manager'].clear_collection.assert_called_once()
        mock_dependencies['doc_processor'].convert_pdfs.assert_called_once()
        mock_dependencies['doc_processor'].chunk_documents.assert_called_once()
        mock_dependencies['vector_manager'].store_documents.assert_called_once()
    
    def test_reprocess_documents_no_documents(self, mock_openai_api_key, mock_dependencies):
        """Test document reprocessing when no documents are found"""
        service = RAGService()
        
        # Mock no documents found
        mock_dependencies['doc_processor'].convert_pdfs.return_value = []
        
        success = service.reprocess_documents()
        
        assert success is False
    
    def test_reprocess_documents_chunking_failure_with_fallback(self, mock_openai_api_key, mock_dependencies):
        """Test document reprocessing with chunking failure and fallback"""
        service = RAGService()
        
        # Mock document processing
        mock_documents = [Document(page_content="Test document", metadata={'source': 'test.pdf'})]
        mock_dependencies['doc_processor'].convert_pdfs.return_value = mock_documents
        
        # Mock chunking failure then success with fallback
        mock_chunks = [Document(page_content="Fallback chunk", metadata={'source': 'test.pdf'})]
        mock_dependencies['doc_processor'].chunk_documents.side_effect = [
            Exception("Chunking failed"),  # First call fails
            mock_chunks  # Second call (fallback) succeeds
        ]
        
        # Mock vector store operations
        mock_dependencies['vector_manager'].clear_collection.return_value = True
        mock_dependencies['vector_manager'].store_documents.return_value = True
        
        success = service.reprocess_documents()
        
        assert success is True
        assert mock_dependencies['doc_processor'].chunk_documents.call_count == 2
    
    def test_reprocess_documents_vector_store_failure(self, mock_openai_api_key, mock_dependencies):
        """Test document reprocessing with vector store failure"""
        service = RAGService()
        
        # Mock document processing
        mock_documents = [Document(page_content="Test document", metadata={'source': 'test.pdf'})]
        mock_dependencies['doc_processor'].convert_pdfs.return_value = mock_documents
        
        # Mock chunking
        mock_chunks = [Document(page_content="Test chunk", metadata={'source': 'test.pdf'})]
        mock_dependencies['doc_processor'].chunk_documents.return_value = mock_chunks
        
        # Mock vector store failure
        mock_dependencies['vector_manager'].clear_collection.return_value = True
        mock_dependencies['vector_manager'].store_documents.return_value = False
        
        success = service.reprocess_documents()
        
        assert success is False
    
    def test_configuration_persistence(self, mock_openai_api_key, mock_dependencies, tmp_path):
        """Test configuration loading and saving"""
        service = RAGService()
        
        # Test saving configuration
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "chunking_config.json"
        
        with patch('rag_service.os.makedirs'), \
             patch('builtins.open', create=True) as mock_open:
            
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            service._save_chunking_config()
            
            mock_open.assert_called_once()
            mock_file.write.assert_called_once()
    
    def test_error_handling_in_query_processing(self, mock_openai_api_key, mock_dependencies):
        """Test comprehensive error handling in query processing"""
        service = RAGService()
        
        # Test retriever creation error
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.side_effect = Exception("Retriever error")
        mock_dependencies['vector_manager'].get_vectorstore.return_value = mock_vectorstore
        
        response = service.query("Test question")
        
        assert isinstance(response, QueryResponse)
        assert response.error is not None
        assert "Error accessing document database" in response.error
    
    def test_concurrent_query_processing(self, mock_openai_api_key, mock_dependencies):
        """Test handling of concurrent query requests"""
        service = RAGService()
        
        # Mock vector store and retriever
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_dependencies['vector_manager'].get_vectorstore.return_value = mock_vectorstore
        
        # Mock QA chain responses
        mock_qa_result = {
            'result': 'Answer',
            'source_documents': [Mock(metadata={'source': 'test.pdf'})]
        }
        
        with patch('rag_service.RetrievalQA') as mock_qa:
            mock_qa_chain = Mock()
            mock_qa_chain.invoke.return_value = mock_qa_result
            mock_qa.from_chain_type.return_value = mock_qa_chain
            
            # Simulate concurrent requests
            responses = []
            for i in range(3):
                response = service.query(f"Question {i}")
                responses.append(response)
        
        # All requests should succeed
        for response in responses:
            assert isinstance(response, QueryResponse)
            assert response.answer == 'Answer'
            assert response.error is None


class TestRAGServiceConfigurationIntegration:
    """Integration tests for configuration management"""
    
    @pytest.fixture
    def mock_openai_api_key(self, monkeypatch):
        """Mock OpenAI API key for testing"""
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    
    @pytest.fixture
    def service_with_mocks(self, mock_openai_api_key):
        """Create RAG service with mocked dependencies"""
        with patch('rag_service.DocumentProcessor'), \
             patch('rag_service.VectorStoreManager'), \
             patch('rag_service.ChatOpenAI'), \
             patch('rag_service.OpenAIEmbeddings'), \
             patch('rag_service.ErrorRecovery'):
            return RAGService()
    
    def test_configuration_workflow_complete(self, service_with_mocks):
        """Test complete configuration update workflow"""
        service = service_with_mocks
        
        # Get initial strategies
        initial_strategies = service.get_chunking_strategies()
        assert 'strategies' in initial_strategies
        assert 'current_config' in initial_strategies
        
        # Update configuration
        new_config = {
            'chunk_size': 1200,
            'chunk_overlap': 120,
            'similarity_threshold': 0.75
        }
        
        with patch.object(service, '_save_chunking_config'):
            success = service.update_chunking_config('semantic', new_config)
        
        assert success is True
        
        # Verify configuration was updated
        updated_strategies = service.get_chunking_strategies()
        current_config = updated_strategies['current_config']
        
        assert current_config['strategy'] == 'semantic'
        assert current_config['chunk_size'] == 1200
        assert current_config['chunk_overlap'] == 120
        assert current_config['similarity_threshold'] == 0.75
    
    def test_configuration_rollback_on_error(self, service_with_mocks):
        """Test configuration rollback when update fails"""
        service = service_with_mocks
        
        # Store original configuration
        original_config = service.chunking_config
        
        # Attempt invalid configuration update
        invalid_config = {'chunk_size': 'invalid_value'}
        success = service.update_chunking_config('recursive_character', invalid_config)
        
        assert success is False
        
        # Verify configuration was not changed
        assert service.chunking_config.strategy == original_config.strategy
        assert service.chunking_config.chunk_size == original_config.chunk_size
    
    def test_multiple_strategy_configurations(self, service_with_mocks):
        """Test configuring different chunking strategies"""
        service = service_with_mocks
        
        # Test configurations for different strategies
        test_configs = [
            ('recursive_character', {'chunk_size': 800, 'chunk_overlap': 80}),
            ('semantic', {'similarity_threshold': 0.9, 'chunk_size': 1000}),
            ('token_based', {'max_tokens': 512, 'tokenizer_model': 'gpt-3.5-turbo'}),
            ('paragraph', {'chunk_size': 1500, 'chunk_overlap': 150}),
            ('hybrid', {'chunk_size': 1000, 'similarity_threshold': 0.8})
        ]
        
        for strategy, config in test_configs:
            with patch.object(service, '_save_chunking_config'):
                success = service.update_chunking_config(strategy, config)
            
            assert success is True, f"Failed to configure {strategy}"
            assert service.chunking_config.strategy == strategy