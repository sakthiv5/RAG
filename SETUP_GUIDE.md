# RAG System Setup Guide

This guide provides detailed instructions for setting up the RAG system with the new web UI functionality.

## System Requirements

- **Python**: 3.10.5 or higher
- **Docker**: Latest version with Docker Compose
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 2GB free space for dependencies and vector data
- **Network**: Internet connection for OpenAI API and package downloads

## Step-by-Step Setup

### 1. Environment Preparation

#### Create Virtual Environment

```bash
# Create virtual environment
python -m venv rag

# Activate virtual environment
# Windows:
rag\Scripts\activate
# Linux/Mac:
source rag/bin/activate

# Verify Python version
python --version  # Should be 3.10.5 or higher
```

#### Install Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify critical packages
python -c "import flask, langchain, docling, tiktoken, sentence_transformers; print('All packages installed successfully')"
```

### 2. Environment Configuration

#### Create Environment File

Create a `.env` file in the project root:

```bash
# Required - Get from https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional - Service Configuration (defaults shown)
MILVUS_HOST=localhost
MILVUS_PORT=19530
MINIO_USERNAME=minioadmin
MINIO_PASSWORD=minioadmin

# Optional - Flask Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=false
FLASK_SECRET_KEY=your-secure-secret-key-for-production

# Optional - Application Settings
LOG_LEVEL=INFO
DATA_DIR=data
CONFIG_DIR=config
```

#### Validate Environment

```bash
# Run environment validation
python validate_env.py

# Expected output:
# ✅ Environment validation passed
# ✅ OpenAI API key is valid
# ✅ All required variables are set
```

### 3. Docker Services Setup

#### Start Required Services

```bash
# Start Milvus, etcd, and MinIO services
docker-compose up -d

# Verify services are running
docker-compose ps

# Expected output should show all services as "Up"
```

#### Verify Service Connectivity

```bash
# Test Milvus connection
python -c "
from vector_store_manager import VectorStoreManager
vm = VectorStoreManager()
print('✅ Milvus connection successful' if vm.test_connection() else '❌ Milvus connection failed')
"
```

### 4. Initial Data Setup

#### Prepare Document Directory

```bash
# Create data directory if it doesn't exist
mkdir -p data

# Add your PDF documents
# Example:
# cp /path/to/your/documents/*.pdf data/
```

#### Verify Document Processing

```bash
# Test document processing (optional)
python -c "
from document_processor import DocumentProcessor
dp = DocumentProcessor()
docs = dp.convert_pdfs()
print(f'✅ Found and processed {len(docs)} documents')
"
```

### 5. Application Startup

#### Start the Web Application

```bash
# Start Flask application
python app.py

# Expected output:
# * Running on http://0.0.0.0:5000
# * Debug mode: off
```

#### Verify Web Interface

1. Open browser to `http://localhost:5000`
2. You should see the RAG system web interface
3. Try asking a test question to verify functionality

## New Dependencies Explained

### Core Web Framework
- **Flask 3.0.0**: Web framework for the user interface
- **Werkzeug 3.0.1**: WSGI toolkit (Flask dependency)
- **Jinja2**: Template engine for HTML rendering
- **MarkupSafe**: Template security utilities

### Enhanced Chunking Capabilities
- **tiktoken**: Token counting for OpenAI models (token-based chunking)
- **sentence-transformers**: Sentence embeddings for semantic chunking
- **scikit-learn**: Machine learning utilities for similarity calculations
- **numpy**: Numerical operations for embeddings

### Existing Dependencies (Enhanced)
- **docling**: PDF to markdown conversion
- **langchain**: LLM application framework
- **langchain-openai**: OpenAI integrations (GPT-4-mini, embeddings)
- **langchain-community**: Community integrations (Milvus)
- **python-dotenv**: Environment variable management

### Development Tools (Optional)
- **pytest**: Testing framework
- **pytest-flask**: Flask-specific testing utilities
- **black**: Code formatting
- **flake8**: Code linting

## Configuration Management

### Default Configuration

The system creates a default chunking configuration on first run:

```json
{
  "strategy": "recursive",
  "chunk_size": 1000,
  "chunk_overlap": 100,
  "separators": ["\n\n", "\n", " ", ""]
}
```

### Configuration CLI

Use the configuration CLI for advanced management:

```bash
# Show current configuration
python config_cli.py config show

# List available strategies
python config_cli.py config list-strategies

# Create backup before changes
python config_cli.py backup create setup

# Reset to defaults if needed
python config_cli.py config reset
```

## Troubleshooting Setup Issues

### Common Installation Problems

#### 1. Python Version Issues
```bash
# Error: Python version too old
# Solution: Install Python 3.10.5 or higher
python --version
```

#### 2. Package Installation Failures
```bash
# Error: Failed building wheel for sentence-transformers
# Solution: Install build tools
pip install --upgrade setuptools wheel
pip install -r requirements.txt
```

#### 3. Docker Service Issues
```bash
# Error: Docker services not starting
# Solution: Check Docker is running and ports are available
docker --version
docker-compose --version
netstat -an | findstr :19530  # Check if Milvus port is free
```

#### 4. OpenAI API Key Issues
```bash
# Error: Invalid API key
# Solution: Verify key format and permissions
python -c "
import openai
from dotenv import load_dotenv
import os
load_dotenv()
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print('✅ API key valid')
"
```

### Service Connectivity Issues

#### Milvus Connection Problems
```bash
# Check Milvus logs
docker-compose logs milvus

# Restart Milvus if needed
docker-compose restart milvus

# Test connection
python -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('✅ Connected')"
```

#### Port Conflicts
```bash
# Check what's using port 5000
netstat -an | findstr :5000

# Change Flask port if needed
export FLASK_PORT=5001
python app.py
```

### Performance Optimization

#### Memory Usage
```bash
# Monitor memory usage during setup
# Sentence transformers models can use significant memory
python -c "
import psutil
print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB')
"
```

#### Model Downloads
```bash
# Pre-download sentence transformer models (optional)
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('✅ Model downloaded and cached')
"
```

## Verification Checklist

After completing setup, verify these components:

- [ ] Python virtual environment activated
- [ ] All dependencies installed without errors
- [ ] `.env` file created with valid OpenAI API key
- [ ] Docker services running (milvus, etcd, minio)
- [ ] Environment validation passes
- [ ] Flask application starts without errors
- [ ] Web interface accessible at `http://localhost:5000`
- [ ] Can ask questions and receive responses
- [ ] Can change chunking strategies
- [ ] Document reprocessing works

## Next Steps

1. **Add Documents**: Place PDF files in the `data/` directory
2. **Configure Chunking**: Experiment with different chunking strategies
3. **Test Queries**: Ask questions to verify the system works
4. **Monitor Performance**: Check response times and accuracy
5. **Backup Configuration**: Create backups of working configurations

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review logs: `docker-compose logs` and Flask console output
3. Validate environment: `python validate_env.py`
4. Reset configuration: `python config_cli.py config reset`
5. Check the main README.md for additional information

## Production Considerations

For production deployment:

1. **Security**: Use strong secret keys and secure API key storage
2. **Performance**: Consider using a production WSGI server like Gunicorn
3. **Monitoring**: Set up logging and health checks
4. **Scaling**: Consider load balancing for multiple users
5. **Backup**: Regular backups of configuration and vector data