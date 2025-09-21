"""
API endpoint tests for Flask routes.
Tests all Flask routes with various input scenarios, error handling, and edge cases.
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from flask import Flask

from app import app, rag_service, reprocessing_status
from rag_service import QueryResponse


class TestFlaskAPIEndpoints:
    """Test Flask API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def mock_rag_service(self):
        """Mock RAG service for testing"""
        with patch('app.rag_service') as mock_service:
            yield mock_service


class TestQueryEndpoint(TestFlaskAPIEndpoints):
    """Test /query endpoint"""
    
    def test_query_success(self, client, mock_rag_service):
        """Test successful query processing"""
        # Mock successful response
        mock_response = QueryResponse(
            answer="This is the answer",
            sources=["document1.pdf", "document2.pdf"],
            processing_time=1.5,
            chunk_count=3,
            error=None
        )
        mock_rag_service.query.return_value = mock_response
        
        response = client.post('/query', 
                             json={'question': 'What is the main topic?'},
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['answer'] == "This is the answer"
        assert data['sources'] == ["document1.pdf", "document2.pdf"]
        assert data['processing_time'] == 1.5
        assert data['chunk_count'] == 3
    
    def test_query_with_error(self, client, mock_rag_service):
        """Test query processing with error"""
        # Mock error response
        mock_response = QueryResponse(
            answer="",
            sources=[],
            processing_time=0.5,
            chunk_count=0,
            error="No documents available"
        )
        mock_rag_service.query.return_value = mock_response
        
        response = client.post('/query',
                             json={'question': 'What is the main topic?'},
                             content_type='application/json')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        
        assert data['error'] == "No documents available"
        assert data['answer'] == ""
        assert data['sources'] == []
    
    def test_query_no_json_data(self, client):
        """Test query endpoint without JSON data"""
        response = client.post('/query')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'JSON data' in data['error']
    
    def test_query_empty_question(self, client):
        """Test query endpoint with empty question"""
        response = client.post('/query',
                             json={'question': ''},
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_query_invalid_question_type(self, client):
        """Test query endpoint with invalid question type"""
        response = client.post('/query',
                             json={'question': 123},
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_query_service_unavailable(self, client):
        """Test query endpoint when RAG service is unavailable"""
        with patch('app.rag_service', None):
            response = client.post('/query',
                                 json={'question': 'Test question'},
                                 content_type='application/json')
        
        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'Service temporarily unavailable' in data['error']
    
    def test_query_service_returns_none(self, client, mock_rag_service):
        """Test query endpoint when service returns None"""
        mock_rag_service.query.return_value = None
        
        response = client.post('/query',
                             json={'question': 'Test question'},
                             content_type='application/json')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'Failed to process query' in data['error']
    
    def test_query_long_question(self, client, mock_rag_service):
        """Test query endpoint with very long question"""
        long_question = "What is " * 1000  # Very long question
        
        mock_response = QueryResponse(
            answer="Answer to long question",
            sources=["doc.pdf"],
            processing_time=2.0,
            chunk_count=1
        )
        mock_rag_service.query.return_value = mock_response
        
        response = client.post('/query',
                             json={'question': long_question},
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['answer'] == "Answer to long question"


class TestChunkingStrategiesEndpoint(TestFlaskAPIEndpoints):
    """Test /chunking-strategies endpoint"""
    
    def test_get_chunking_strategies_success(self, client, mock_rag_service):
        """Test successful retrieval of chunking strategies"""
        mock_strategies_data = {
            'strategies': {
                'recursive_character': {
                    'name': 'Recursive Character Text Splitter',
                    'description': 'Default recursive character text splitter',
                    'config_options': ['chunk_size', 'chunk_overlap']
                }
            },
            'current_config': {
                'strategy': 'recursive_character',
                'chunk_size': 1000,
                'chunk_overlap': 100
            }
        }
        mock_rag_service.get_chunking_strategies.return_value = mock_strategies_data
        
        response = client.get('/chunking-strategies')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'strategies' in data
        assert 'current_strategy' in data
        assert 'current_config' in data
        assert data['current_strategy'] == 'recursive_character'
    
    def test_get_chunking_strategies_service_unavailable(self, client):
        """Test chunking strategies endpoint when service is unavailable"""
        with patch('app.rag_service', None):
            response = client.get('/chunking-strategies')
        
        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'Service temporarily unavailable' in data['error']
    
    def test_get_chunking_strategies_service_error(self, client, mock_rag_service):
        """Test chunking strategies endpoint when service returns None"""
        mock_rag_service.get_chunking_strategies.return_value = None
        
        response = client.get('/chunking-strategies')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'Failed to retrieve chunking strategies' in data['error']


class TestChunkingConfigEndpoint(TestFlaskAPIEndpoints):
    """Test /chunking-config endpoint"""
    
    def test_update_chunking_config_success(self, client, mock_rag_service):
        """Test successful chunking configuration update"""
        mock_rag_service.update_chunking_config.return_value = True
        mock_rag_service.get_chunking_strategies.return_value = {
            'current_config': {
                'strategy': 'recursive_character',
                'chunk_size': 800,
                'chunk_overlap': 80
            }
        }
        
        config_data = {
            'strategy': 'recursive_character',
            'config': {
                'chunk_size': 800,
                'chunk_overlap': 80
            }
        }
        
        response = client.post('/chunking-config',
                             json=config_data,
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'message' in data
        assert 'Configuration updated successfully' in data['message']
        assert 'current_config' in data
    
    def test_update_chunking_config_no_json(self, client):
        """Test chunking config update without JSON data"""
        response = client.post('/chunking-config')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Request must contain configuration data' in data['error']
    
    def test_update_chunking_config_no_strategy(self, client):
        """Test chunking config update without strategy"""
        response = client.post('/chunking-config',
                             json={'config': {'chunk_size': 800}},
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Chunking strategy is required' in data['error']
    
    def test_update_chunking_config_invalid_config(self, client):
        """Test chunking config update with invalid configuration"""
        config_data = {
            'strategy': 'recursive_character',
            'config': {
                'chunk_size': -100  # Invalid chunk size
            }
        }
        
        response = client.post('/chunking-config',
                             json=config_data,
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Invalid configuration' in data['error']
    
    def test_update_chunking_config_service_failure(self, client, mock_rag_service):
        """Test chunking config update when service fails"""
        mock_rag_service.update_chunking_config.return_value = False
        
        config_data = {
            'strategy': 'recursive_character',
            'config': {'chunk_size': 800}
        }
        
        response = client.post('/chunking-config',
                             json=config_data,
                             content_type='application/json')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'Failed to update configuration' in data['error']
    
    def test_update_chunking_config_service_unavailable(self, client):
        """Test chunking config update when service is unavailable"""
        with patch('app.rag_service', None):
            config_data = {
                'strategy': 'recursive_character',
                'config': {'chunk_size': 800}
            }
            
            response = client.post('/chunking-config',
                                 json=config_data,
                                 content_type='application/json')
        
        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'Service temporarily unavailable' in data['error']


class TestReprocessEndpoints(TestFlaskAPIEndpoints):
    """Test /reprocess and /reprocess-status endpoints"""
    
    def setUp(self):
        """Reset reprocessing status before each test"""
        global reprocessing_status
        reprocessing_status.update({
            'status': 'idle',
            'progress': 0,
            'message': '',
            'documents_processed': 0,
            'chunks_created': 0
        })
    
    def test_reprocess_documents_success(self, client, mock_rag_service):
        """Test successful document reprocessing"""
        self.setUp()
        mock_rag_service.reprocess_documents.return_value = True
        
        response = client.post('/reprocess')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'message' in data
        assert 'completed successfully' in data['message']
        assert data['status']['status'] == 'completed'
        assert data['status']['progress'] == 100
    
    def test_reprocess_documents_failure(self, client, mock_rag_service):
        """Test document reprocessing failure"""
        self.setUp()
        mock_rag_service.reprocess_documents.side_effect = Exception("Reprocessing failed")
        
        response = client.post('/reprocess')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        
        assert 'error' in data
        assert data['status']['status'] == 'error'
    
    def test_reprocess_documents_already_processing(self, client):
        """Test reprocessing when already in progress"""
        # Set status to processing
        global reprocessing_status
        reprocessing_status['status'] = 'processing'
        
        response = client.post('/reprocess')
        
        assert response.status_code == 409
        data = json.loads(response.data)
        assert 'already in progress' in data['error']
    
    def test_reprocess_documents_service_unavailable(self, client):
        """Test reprocessing when service is unavailable"""
        self.setUp()
        with patch('app.rag_service', None):
            response = client.post('/reprocess')
        
        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'Service temporarily unavailable' in data['error']
    
    def test_reprocess_status_idle(self, client):
        """Test reprocess status when idle"""
        self.setUp()
        
        response = client.get('/reprocess-status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'idle'
        assert data['progress'] == 0
        assert data['documents_processed'] == 0
        assert data['chunks_created'] == 0
    
    def test_reprocess_status_processing(self, client):
        """Test reprocess status during processing"""
        # Set status to processing
        global reprocessing_status
        reprocessing_status.update({
            'status': 'processing',
            'progress': 50,
            'message': 'Processing documents...',
            'documents_processed': 2,
            'chunks_created': 10
        })
        
        response = client.get('/reprocess-status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'processing'
        assert data['progress'] == 50
        assert data['message'] == 'Processing documents...'
        assert data['documents_processed'] == 2
        assert data['chunks_created'] == 10


class TestHealthEndpoints(TestFlaskAPIEndpoints):
    """Test health check endpoints"""
    
    def test_basic_health_check(self, client):
        """Test basic health check endpoint"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'healthy'
        assert data['service'] == 'rag-system'
        assert 'timestamp' in data
    
    def test_detailed_health_check_healthy(self, client, mock_rag_service):
        """Test detailed health check when all components are healthy"""
        # Mock healthy components
        mock_rag_service.vector_manager.test_connection.return_value = True
        
        with patch('app.env_manager') as mock_env_manager:
            mock_env_manager.get_validation_report.return_value = {'all_valid': True}
            
            response = client.get('/health/detailed')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'healthy'
        assert 'components' in data
        assert data['components']['rag_service'] == 'healthy'
    
    def test_detailed_health_check_degraded(self, client):
        """Test detailed health check when service is unavailable"""
        with patch('app.rag_service', None):
            response = client.get('/health/detailed')
        
        assert response.status_code == 503
        data = json.loads(response.data)
        
        assert data['status'] == 'degraded'
        assert data['components']['rag_service'] == 'unhealthy'


class TestIndexEndpoint(TestFlaskAPIEndpoints):
    """Test main index endpoint"""
    
    def test_index_page(self, client):
        """Test main index page renders correctly"""
        response = client.get('/')
        
        assert response.status_code == 200
        assert b'RAG Document Q&A System' in response.data
        assert b'Ask a Question' in response.data
        assert b'Chunking Strategy Configuration' in response.data


class TestErrorHandling(TestFlaskAPIEndpoints):
    """Test global error handling"""
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get('/nonexistent-endpoint')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'Endpoint not found' in data['error']
    
    def test_500_error_handling(self, client):
        """Test 500 error handling"""
        with patch('app.render_template', side_effect=Exception("Template error")):
            response = client.get('/')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data


class TestConcurrentRequests(TestFlaskAPIEndpoints):
    """Test handling of concurrent requests"""
    
    def test_concurrent_query_requests(self, client, mock_rag_service):
        """Test handling multiple concurrent query requests"""
        mock_response = QueryResponse(
            answer="Concurrent answer",
            sources=["doc.pdf"],
            processing_time=1.0,
            chunk_count=1
        )
        mock_rag_service.query.return_value = mock_response
        
        # Simulate concurrent requests
        responses = []
        for i in range(5):
            response = client.post('/query',
                                 json={'question': f'Question {i}'},
                                 content_type='application/json')
            responses.append(response)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['answer'] == "Concurrent answer"
    
    def test_concurrent_config_updates(self, client, mock_rag_service):
        """Test handling concurrent configuration updates"""
        mock_rag_service.update_chunking_config.return_value = True
        mock_rag_service.get_chunking_strategies.return_value = {
            'current_config': {'strategy': 'recursive_character'}
        }
        
        config_data = {
            'strategy': 'recursive_character',
            'config': {'chunk_size': 800}
        }
        
        # Simulate concurrent config updates
        responses = []
        for i in range(3):
            response = client.post('/chunking-config',
                                 json=config_data,
                                 content_type='application/json')
            responses.append(response)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200


class TestInputValidation(TestFlaskAPIEndpoints):
    """Test input validation across endpoints"""
    
    def test_query_input_validation(self, client):
        """Test query input validation"""
        # Test various invalid inputs
        invalid_inputs = [
            {'question': None},
            {'question': 123},
            {'question': []},
            {'question': {}},
            {'not_question': 'test'}
        ]
        
        for invalid_input in invalid_inputs:
            response = client.post('/query',
                                 json=invalid_input,
                                 content_type='application/json')
            assert response.status_code == 400
    
    def test_config_input_validation(self, client):
        """Test configuration input validation"""
        # Test various invalid configurations
        invalid_configs = [
            {'strategy': None, 'config': {}},
            {'strategy': 123, 'config': {}},
            {'strategy': 'invalid_strategy', 'config': {}},
            {'config': {'chunk_size': 'invalid'}},
            {'strategy': 'recursive_character', 'config': {'chunk_size': -1}}
        ]
        
        for invalid_config in invalid_configs:
            response = client.post('/chunking-config',
                                 json=invalid_config,
                                 content_type='application/json')
            assert response.status_code == 400