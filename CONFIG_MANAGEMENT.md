# Configuration Management Guide

This document describes the comprehensive configuration management system for the RAG application, including environment variables, chunking strategies, and backup/restore functionality.

## Overview

The RAG application uses a two-tier configuration system:

1. **Environment Variables** - System-level configuration (API keys, service endpoints, etc.)
2. **Chunking Configuration** - Application-level settings for document processing strategies

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4-mini and embeddings | `sk-your-api-key-here` |

### Optional Variables (with defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_HOST` | `localhost` | Milvus vector database host |
| `MILVUS_PORT` | `19530` | Milvus vector database port |
| `MINIO_USERNAME` | `minioadmin` | MinIO username for Milvus storage |
| `MINIO_PASSWORD` | `minioadmin` | MinIO password for Milvus storage |
| `FLASK_SECRET_KEY` | `dev-secret-key` | Flask application secret key |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |
| `FLASK_HOST` | `0.0.0.0` | Flask application host |
| `FLASK_PORT` | `5000` | Flask application port |
| `LOG_LEVEL` | `INFO` | Application log level |
| `DATA_DIR` | `data` | Directory containing PDF documents |
| `CONFIG_DIR` | `config` | Directory for configuration files |

### Environment Setup

1. **Create .env file:**
   ```bash
   # Copy the template
   cp .env.template .env
   
   # Or create from scratch
   echo "OPENAI_API_KEY=sk-your-actual-key-here" > .env
   ```

2. **Validate environment:**
   ```bash
   python validate_env.py
   ```

3. **Create environment template:**
   ```bash
   python config_cli.py env template
   ```

## Chunking Configuration

### Available Strategies

#### 1. Recursive Character Text Splitter (default)
- **ID:** `recursive`
- **Description:** Splits text recursively using a hierarchy of separators
- **Parameters:**
  - `chunk_size`: Maximum chunk size in characters (100-10000, default: 1000)
  - `chunk_overlap`: Overlapping characters between chunks (0-500, default: 100)
  - `separators`: List of separators in preference order

#### 2. Semantic Chunking
- **ID:** `semantic`
- **Description:** Groups sentences based on semantic similarity
- **Parameters:**
  - `similarity_threshold`: Minimum similarity score (0.1-1.0, default: 0.8)
  - `min_chunk_size`: Minimum chunk size in characters (50-1000, default: 200)

#### 3. Token-based Chunking
- **ID:** `token`
- **Description:** Splits text based on token count using tiktoken
- **Parameters:**
  - `max_tokens`: Maximum tokens per chunk (50-8000, default: 1000)
  - `tokenizer_model`: Model for tokenization (gpt-4, gpt-3.5-turbo, etc.)

#### 4. Paragraph-based Chunking
- **ID:** `paragraph`
- **Description:** Splits text on paragraph boundaries
- **Parameters:**
  - `min_chunk_size`: Minimum chunk size (50-1000, default: 200)
  - `max_chunk_size`: Maximum chunk size (500-5000, default: 2000)

#### 5. Hybrid Chunking
- **ID:** `hybrid`
- **Description:** Combines multiple chunking strategies
- **Parameters:**
  - `chunk_size`: Base chunk size for recursive splitting
  - `similarity_threshold`: Similarity threshold for semantic grouping
  - `max_tokens`: Maximum tokens per final chunk

## Configuration Management CLI

The `config_cli.py` tool provides comprehensive configuration management:

### Configuration Commands

```bash
# Show current configuration
python config_cli.py config show

# List available strategies
python config_cli.py config list-strategies

# Reset to default configuration
python config_cli.py config reset
```

### Backup Management

```bash
# List all backups
python config_cli.py backup list

# Create a backup
python config_cli.py backup create [suffix]

# Restore from backup
python config_cli.py backup restore <filename>

# Clean up old backups (keep 5 most recent)
python config_cli.py backup cleanup 5
```

### Import/Export

```bash
# Export configuration
python config_cli.py export config.json

# Import configuration
python config_cli.py import config.json
```

### Environment Management

```bash
# Validate environment
python config_cli.py env validate

# Create .env template
python config_cli.py env template
```

## Programmatic Configuration Management

### Using ConfigManager

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
success = config_manager.update_config(new_config)

# Create backup
backup_path = config_manager.create_backup('manual')

# List backups
backups = config_manager.list_backups()

# Restore from backup
success = config_manager.restore_from_backup('backup_file.json')
```

### Using EnvironmentManager

```python
from env_manager import env_manager, validate_startup_environment

# Validate environment on startup
if not validate_startup_environment():
    sys.exit(1)

# Get validation report
report = env_manager.get_validation_report()

# Check service connectivity
services = env_manager.check_critical_services()
```

## Configuration Files

### File Structure

```
config/
├── chunking_config.json          # Current configuration
├── backups/                      # Configuration backups
│   ├── chunking_config_20240101_120000_manual.json
│   └── chunking_config_20240101_130000_auto.json
└── .gitkeep                      # Keep directory in git
```

### Configuration File Format

```json
{
  "strategy": "semantic",
  "chunk_size": 800,
  "chunk_overlap": 100,
  "similarity_threshold": 0.75,
  "min_chunk_size": 200
}
```

### Backup File Format

```json
{
  "exported_at": "2024-01-01T12:00:00",
  "version": "1.0",
  "config": {
    "strategy": "semantic",
    "chunk_size": 800,
    "chunk_overlap": 100,
    "similarity_threshold": 0.75
  }
}
```

## Automatic Features

### Auto-Backup
- Configuration changes automatically create backups
- Backups are timestamped and include operation context
- Old backups are automatically cleaned up (configurable retention)

### Validation
- Environment variables are validated on startup
- Configuration parameters are validated before saving
- Clear error messages guide users to fix issues

### Default Handling
- Missing optional environment variables use sensible defaults
- Invalid configurations fall back to defaults with warnings
- First-time setup creates default configuration automatically

## Troubleshooting

### Common Issues

1. **Invalid OpenAI API Key**
   ```
   Error: Invalid value for environment variable: OPENAI_API_KEY
   Solution: Get a valid API key from https://platform.openai.com/api-keys
   ```

2. **Service Connectivity Issues**
   ```
   Error: Milvus not accessible
   Solution: Start Docker services with: docker-compose up -d
   ```

3. **Configuration Validation Errors**
   ```
   Error: Invalid chunking configuration
   Solution: Use config_cli.py to reset or validate parameters
   ```

### Recovery Procedures

1. **Reset to defaults:**
   ```bash
   python config_cli.py config reset
   ```

2. **Restore from backup:**
   ```bash
   python config_cli.py backup list
   python config_cli.py backup restore <filename>
   ```

3. **Validate environment:**
   ```bash
   python validate_env.py
   ```

## Best Practices

1. **Environment Variables:**
   - Never commit `.env` files to version control
   - Use strong secret keys in production
   - Validate environment before deployment

2. **Configuration Management:**
   - Create backups before major changes
   - Test configuration changes in development first
   - Document custom configurations for your use case

3. **Backup Management:**
   - Regularly clean up old backups
   - Keep important configurations as named backups
   - Export configurations for sharing between environments

4. **Security:**
   - Rotate API keys regularly
   - Use environment-specific configurations
   - Limit access to configuration files in production