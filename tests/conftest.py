"""
Test configuration and fixtures for pytest.
Sets up test environment and mocks for all tests.
"""
import pytest
import os
from unittest.mock import Mock, patch
import sys

# Mock environment variables before any imports
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables"""
    os.environ["OPENAI_API_KEY"] = "test-api-key-for-testing"
    os.environ["MINIO_USERNAME"] = "testuser"
    os.environ["MINIO_PASSWORD"] = "testpass"
    os.environ["FLASK_DEBUG"] = "false"
    os.environ["LOG_LEVEL"] = "ERROR"  # Reduce logging during tests

# Mock the environment validation to always pass during tests
@pytest.fixture(scope="session", autouse=True)
def mock_environment_validation():
    """Mock environment validation for tests"""
    with patch('env_manager.validate_startup_environment', return_value=True):
        yield

# Mock external dependencies that require network/API calls
@pytest.fixture(scope="session", autouse=True)
def mock_external_dependencies():
    """Mock external dependencies for all tests"""
    with patch('rag_service.ChatOpenAI') as mock_llm, \
         patch('rag_service.OpenAIEmbeddings') as mock_embeddings, \
         patch('document_processor.DocumentProcessor') as mock_doc_processor, \
         patch('vector_store_manager.VectorStoreManager') as mock_vector_manager, \
         patch('rag_service.ErrorRecovery') as mock_error_recovery:
        
        # Setup mock returns
        mock_llm.return_value = Mock()
        mock_embeddings.return_value = Mock()
        mock_doc_processor.return_value = Mock()
        mock_vector_manager.return_value = Mock()
        mock_error_recovery.retry_with_backoff.side_effect = lambda func, **kwargs: func()
        
        yield {
            'llm': mock_llm,
            'embeddings': mock_embeddings,
            'doc_processor': mock_doc_processor,
            'vector_manager': mock_vector_manager,
            'error_recovery': mock_error_recovery
        }

@pytest.fixture
def app():
    """Create test Flask app"""
    # Import after environment is set up
    from app import app as flask_app
    flask_app.config['TESTING'] = True
    return flask_app

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()