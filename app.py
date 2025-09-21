from flask import Flask, render_template, request, jsonify
import os
import sys
from dotenv import load_dotenv
from rag_service import RAGService
from env_manager import validate_startup_environment
from error_handler import error_handler, handle_errors, InputValidator, safe_execute

# Load environment variables
load_dotenv()

# Validate environment on startup
if not validate_startup_environment():
    print("\n❌ Environment validation failed. Please fix the issues above and restart.")
    sys.exit(1)

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'false').lower() in ['true', '1', 'yes']

# Initialize RAG service with error handling
rag_service = None
try:
    print("🚀 Initializing RAG service...")
    rag_service = RAGService()
    print("✅ RAG service initialized successfully!")
except Exception as e:
    if not error_handler.handle_startup_error(e, "rag_service"):
        print("\n❌ Critical startup error. Exiting application.")
        sys.exit(1)

# Global error handler for Flask
@app.errorhandler(404)
def not_found_error(error):
    error_handler.logger.warning(f"404 error: {request.url}")
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    error_handler.logger.error(f"500 error: {str(error)}")
    return jsonify({'error': 'Internal server error occurred'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    error_msg = error_handler.log_error(e, "flask_request", "An unexpected error occurred")
    return jsonify({'error': error_msg}), 500

@app.route('/')
def index():
    """Serve the main UI page"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
@handle_errors("query_processing")
def query():
    """Process user questions"""
    # Check if RAG service is available
    if not rag_service:
        error_handler.logger.error("RAG service not available for query processing")
        return jsonify({
            'error': 'Service temporarily unavailable. Please try again later.'
        }), 503
    
    # Get and validate JSON data
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request must contain JSON data'}), 400
    
    # Validate and sanitize question
    try:
        question = InputValidator.validate_question(data.get('question', ''))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    # Log the query attempt
    error_handler.logger.info(f"Processing query: {question[:100]}...")
    
    # Process the query with error handling
    response = safe_execute(
        rag_service.query,
        question,
        default_return=None,
        context="query_processing"
    )
    
    if response is None:
        return jsonify({
            'error': 'Failed to process query. Please try again.',
            'answer': '',
            'sources': [],
            'processing_time': 0.0,
            'chunk_count': 0
        }), 500
    
    # Check for errors in response
    if hasattr(response, 'error') and response.error:
        error_handler.logger.warning(f"Query processing error: {response.error}")
        return jsonify({
            'error': response.error,
            'answer': '',
            'sources': [],
            'processing_time': getattr(response, 'processing_time', 0.0),
            'chunk_count': 0
        }), 500
    
    # Log successful query
    error_handler.logger.info(f"Query processed successfully in {response.processing_time:.2f}s")
    
    # Return successful response
    return jsonify({
        'answer': response.answer,
        'sources': response.sources,
        'processing_time': response.processing_time,
        'chunk_count': response.chunk_count
    })

@app.route('/chunking-strategies', methods=['GET'])
@handle_errors("get_chunking_strategies")
def get_chunking_strategies():
    """Get available strategies and current config"""
    # Check if RAG service is available
    if not rag_service:
        return jsonify({
            'error': 'Service temporarily unavailable. Please try again later.'
        }), 503
    
    # Get strategies with error handling
    strategies_data = safe_execute(
        rag_service.get_chunking_strategies,
        default_return=None,
        context="get_chunking_strategies"
    )
    
    if strategies_data is None:
        return jsonify({'error': 'Failed to retrieve chunking strategies'}), 500
    
    return jsonify({
        'strategies': strategies_data['strategies'],
        'current_strategy': strategies_data['current_config']['strategy'],
        'current_config': strategies_data['current_config']
    })

@app.route('/chunking-config', methods=['POST'])
@handle_errors("update_chunking_config")
def update_chunking_config():
    """Update chunking configuration"""
    # Check if RAG service is available
    if not rag_service:
        return jsonify({
            'error': 'Service temporarily unavailable. Please try again later.'
        }), 503
    
    # Get and validate JSON data
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request must contain configuration data'}), 400
    
    # Extract and validate strategy and config
    strategy = data.get('strategy')
    config = data.get('config', {})
    
    if not strategy:
        return jsonify({'error': 'Chunking strategy is required'}), 400
    
    # Validate configuration
    try:
        validated_config = InputValidator.validate_chunking_config(strategy, config)
    except ValueError as e:
        return jsonify({'error': f'Invalid configuration: {str(e)}'}), 400
    
    # Log configuration update attempt
    error_handler.logger.info(f"Updating chunking config: strategy={strategy}")
    
    # Update configuration with error handling
    success = safe_execute(
        rag_service.update_chunking_config,
        strategy,
        validated_config,
        default_return=False,
        context="update_chunking_config"
    )
    
    if success:
        # Get updated configuration
        updated_data = safe_execute(
            rag_service.get_chunking_strategies,
            default_return=None,
            context="get_updated_config"
        )
        
        if updated_data:
            error_handler.logger.info(f"Chunking configuration updated successfully: {strategy}")
            return jsonify({
                'message': 'Configuration updated successfully',
                'current_config': updated_data['current_config']
            })
    
    return jsonify({'error': 'Failed to update configuration. Please check your settings and try again.'}), 500

# Global variable to track reprocessing status
reprocessing_status = {
    'status': 'idle',  # 'idle', 'processing', 'completed', 'error'
    'progress': 0,
    'message': '',
    'documents_processed': 0,
    'chunks_created': 0
}

@app.route('/reprocess', methods=['POST'])
@handle_errors("reprocess_documents")
def reprocess_documents():
    """Trigger document reprocessing"""
    global reprocessing_status
    
    # Check if RAG service is available
    if not rag_service:
        return jsonify({
            'error': 'Service temporarily unavailable. Please try again later.'
        }), 503
    
    # Check if already processing
    if reprocessing_status['status'] == 'processing':
        error_handler.logger.warning("Reprocessing request rejected - already in progress")
        return jsonify({
            'error': 'Document reprocessing is already in progress. Please wait for it to complete.'
        }), 409
    
    # Log reprocessing start
    error_handler.logger.info("Starting document reprocessing")
    
    # Reset status
    reprocessing_status.update({
        'status': 'processing',
        'progress': 0,
        'message': 'Starting document reprocessing...',
        'documents_processed': 0,
        'chunks_created': 0
    })
    
    # Start reprocessing with comprehensive error handling
    try:
        reprocessing_status['message'] = 'Reprocessing documents with current chunking strategy...'
        reprocessing_status['progress'] = 25
        
        # Execute reprocessing with retry logic
        from error_handler import ErrorRecovery
        
        def reprocess_with_retry():
            return rag_service.reprocess_documents()
        
        success = ErrorRecovery.retry_with_backoff(
            reprocess_with_retry,
            max_retries=2,
            backoff_factor=1.0
        )
        
        if success:
            reprocessing_status.update({
                'status': 'completed',
                'progress': 100,
                'message': 'Document reprocessing completed successfully',
                'documents_processed': 1,  # This could be enhanced to track actual count
                'chunks_created': 0  # This could be enhanced to track actual count
            })
            
            error_handler.logger.info("Document reprocessing completed successfully")
            
            return jsonify({
                'message': 'Document reprocessing completed successfully',
                'status': reprocessing_status
            })
        else:
            raise RuntimeError("Reprocessing failed after retries")
            
    except Exception as e:
        error_msg = error_handler.log_error(e, "document_reprocessing")
        
        reprocessing_status.update({
            'status': 'error',
            'progress': 0,
            'message': error_msg,
            'documents_processed': 0,
            'chunks_created': 0
        })
        
        return jsonify({
            'error': error_msg,
            'status': reprocessing_status
        }), 500

@app.route('/reprocess-status', methods=['GET'])
@handle_errors("get_reprocess_status")
def get_reprocess_status():
    """Check reprocessing progress"""
    return jsonify({
        'status': reprocessing_status['status'],
        'progress': reprocessing_status['progress'],
        'message': reprocessing_status['message'],
        'documents_processed': reprocessing_status['documents_processed'],
        'chunks_created': reprocessing_status['chunks_created']
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'rag-system',
        'timestamp': error_handler.get_timestamp()
    }), 200

@app.route('/health/detailed', methods=['GET'])
@handle_errors("detailed_health_check")
def detailed_health_check():
    """Detailed health check with service status"""
    health_status = {
        'status': 'healthy',
        'service': 'rag-system',
        'timestamp': error_handler.get_timestamp(),
        'components': {}
    }
    
    # Check RAG service
    if rag_service:
        health_status['components']['rag_service'] = 'healthy'
    else:
        health_status['components']['rag_service'] = 'unhealthy'
        health_status['status'] = 'degraded'
    
    # Check vector store connection
    try:
        if rag_service and hasattr(rag_service, 'vector_manager'):
            # Test vector store connection
            connection_test = safe_execute(
                rag_service.vector_manager.test_connection,
                default_return=False,
                context="health_check_vector_store"
            )
            health_status['components']['vector_store'] = 'healthy' if connection_test else 'unhealthy'
            if not connection_test:
                health_status['status'] = 'degraded'
        else:
            health_status['components']['vector_store'] = 'unknown'
    except Exception:
        health_status['components']['vector_store'] = 'unhealthy'
        health_status['status'] = 'degraded'
    
    # Check environment configuration
    try:
        from env_manager import env_manager
        env_report = env_manager.get_validation_report()
        if env_report.get('all_valid', False):
            health_status['components']['environment'] = 'healthy'
        else:
            health_status['components']['environment'] = 'unhealthy'
            health_status['status'] = 'degraded'
    except Exception:
        health_status['components']['environment'] = 'unknown'
    
    # Determine overall status code
    status_code = 200 if health_status['status'] == 'healthy' else 503
    
    return jsonify(health_status), status_code

if __name__ == '__main__':
    # Get configuration from environment
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() in ['true', '1', 'yes']
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', '5000'))
    
    print(f"🌐 Starting Flask application on {host}:{port}")
    print(f"🔧 Debug mode: {debug_mode}")
    
    app.run(debug=debug_mode, host=host, port=port)