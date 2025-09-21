# RAG System with Web UI

A Retrieval-Augmented Generation (RAG) system that processes PDF documents and enables intelligent querying through a web-based interface. The system uses OpenAI GPT-4-mini for question answering and provides multiple chunking strategies for optimal document processing.

## Features

- **Web-based UI**: Simple, responsive interface for asking questions and managing configurations
- **Multiple Chunking Strategies**: Choose from recursive character, semantic, token-based, paragraph-based, and hybrid chunking
- **OpenAI Integration**: Uses GPT-4-mini for high-quality responses with source attribution
- **Dynamic Reconfiguration**: Change chunking strategies and reprocess documents without restarting
- **Vector Search**: Powered by Milvus vector database for efficient document retrieval
- **PDF Processing**: Automatic conversion of PDF documents to searchable text using Docling

## Quick Start

### Prerequisites

- Python 3.10.5 or higher
- Docker and Docker Compose
- OpenAI API key

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd rag-system

# Create virtual environment
python -m venv rag
rag\Scripts\activate  # Windows
# source rag/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```bash
# Required
OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional (with defaults)
MILVUS_HOST=localhost
MILVUS_PORT=19530
MINIO_USERNAME=minioadmin
MINIO_PASSWORD=minioadmin
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=false
```

Validate your environment:

```bash
python validate_env.py
```

### 3. Start Services

Start the required Docker services:

```bash
docker-compose up -d
```

Verify services are running:

```bash
docker-compose ps
```

### 4. Add Documents

Place your PDF documents in the `data/` directory:

```bash
mkdir -p data
# Copy your PDF files to the data directory
```

### 5. Start the Web Application

```bash
python app.py
```

The web interface will be available at `http://localhost:5000`

## Using the Web Interface

### Asking Questions

1. Open your browser to `http://localhost:5000`
2. Type your question in the text input field
3. Click "Ask Question" or press Enter
4. View the AI-generated answer with source document references

### Changing Chunking Strategies

1. Use the "Chunking Strategy" dropdown to select a strategy:
   - **Recursive Character** (default): Splits text using hierarchical separators
   - **Semantic Chunking**: Groups text based on semantic similarity
   - **Token-based**: Respects token limits for LLM compatibility
   - **Paragraph-based**: Splits on paragraph boundaries
   - **Hybrid**: Combines multiple strategies

2. Configure strategy-specific parameters in the configuration panel
3. Click "Save Configuration" to apply changes
4. Use "Reprocess Documents" to update the vector database with new chunks

### Reprocessing Documents

When you change chunking strategies or add new documents:

1. Click the "Reprocess Documents" button
2. Confirm the action in the dialog
3. Monitor the progress indicator
4. Wait for completion confirmation

## Available Chunking Strategies

### Recursive Character Text Splitter (Default)
- **Best for**: General-purpose document processing
- **Parameters**:
  - Chunk Size: 100-10,000 characters (default: 1000)
  - Chunk Overlap: 0-500 characters (default: 100)
  - Separators: Custom separator hierarchy

### Semantic Chunking
- **Best for**: Documents where context and meaning are crucial
- **Parameters**:
  - Similarity Threshold: 0.1-1.0 (default: 0.8)
  - Minimum Chunk Size: 50-1000 characters (default: 200)

### Token-based Chunking
- **Best for**: Ensuring compatibility with LLM token limits
- **Parameters**:
  - Max Tokens: 50-8000 tokens (default: 1000)
  - Tokenizer Model: gpt-4, gpt-3.5-turbo, etc.

### Paragraph-based Chunking
- **Best for**: Documents with clear paragraph structure
- **Parameters**:
  - Minimum Chunk Size: 50-1000 characters (default: 200)
  - Maximum Chunk Size: 500-5000 characters (default: 2000)

### Hybrid Chunking
- **Best for**: Complex documents requiring multiple approaches
- **Parameters**:
  - Base Chunk Size: For initial recursive splitting
  - Similarity Threshold: For semantic grouping
  - Max Tokens: Final token limit enforcement

## Configuration Management

### Command Line Interface

The system includes a comprehensive CLI for configuration management:

```bash
# Show current configuration
python config_cli.py config show

# List available strategies
python config_cli.py config list-strategies

# Reset to defaults
python config_cli.py config reset

# Create configuration backup
python config_cli.py backup create

# List backups
python config_cli.py backup list

# Restore from backup
python config_cli.py backup restore <filename>

# Validate environment
python config_cli.py env validate
```

### Programmatic Configuration

```python
from config import config_manager

# Get current configuration
config = config_manager.get_current_config()

# Update configuration
new_config = {
    'strategy': 'semantic',
    'similarity_threshold': 0.75,
    'min_chunk_size': 300
}
config_manager.update_config(new_config)
```

## API Endpoints

The Flask application provides REST API endpoints:

- `GET /` - Web interface
- `POST /query` - Process questions
- `GET /chunking-strategies` - Get available strategies
- `POST /chunking-config` - Update chunking configuration
- `POST /reprocess` - Trigger document reprocessing
- `GET /reprocess-status` - Check reprocessing progress

## Development

### Project Structure

```
├── app.py                     # Flask web application
├── rag_service.py            # Main RAG orchestration
├── document_processor.py     # PDF processing and chunking
├── vector_store_manager.py   # Milvus vector database management
├── chunking_strategies.py    # Chunking strategy implementations
├── config.py                 # Configuration management
├── env_manager.py           # Environment validation
├── error_handler.py         # Error handling utilities
├── templates/               # HTML templates
│   └── index.html          # Main web interface
├── static/                 # Static web assets
│   ├── css/style.css      # Styling
│   └── js/app.js          # JavaScript functionality
├── data/                   # PDF documents
├── config/                 # Configuration files
└── tests/                  # Test suite
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-flask

# Run all tests
pytest

# Run specific test file
pytest tests/test_chunking_strategies.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Issues**
   ```
   Error: Invalid OpenAI API key
   Solution: Verify your API key in the .env file
   ```

2. **Docker Services Not Running**
   ```
   Error: Connection to Milvus failed
   Solution: Start services with: docker-compose up -d
   ```

3. **Port Already in Use**
   ```
   Error: Port 5000 is already in use
   Solution: Change FLASK_PORT in .env or stop conflicting service
   ```

4. **No Documents Found**
   ```
   Error: No PDF documents in data directory
   Solution: Add PDF files to the data/ directory
   ```

### Service Status

Check if all services are running:

```bash
# Check Docker services
docker-compose ps

# Check Flask application
curl http://localhost:5000

# Check Milvus connection
python -c "from vector_store_manager import VectorStoreManager; VectorStoreManager().test_connection()"
```

### Logs and Debugging

```bash
# View Docker service logs
docker-compose logs milvus
docker-compose logs etcd
docker-compose logs minio

# Enable Flask debug mode
export FLASK_DEBUG=true
python app.py
```

## Production Deployment

### Environment Variables

Set production-appropriate values:

```bash
FLASK_DEBUG=false
FLASK_SECRET_KEY=your-secure-secret-key
LOG_LEVEL=WARNING
```

### Docker Deployment

The application can be containerized for production deployment. Update `docker-compose.yml` to include the Flask application service.

### Health Checks

The application includes health check endpoints for monitoring:

- `/health` - Basic health check
- `/health/detailed` - Detailed service status

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the configuration management guide
3. Check existing issues in the repository
4. Create a new issue with detailed information

## Changelog

### Version 2.0.0
- Added web-based user interface
- Integrated OpenAI GPT-4-mini
- Multiple chunking strategies
- Dynamic configuration management
- Document reprocessing capabilities

### Version 1.0.0
- Initial command-line RAG system
- PDF processing with Docling
- Milvus vector storage
- Basic question-answering