"""
Comprehensive error handling and logging system for the RAG application.
Provides centralized error handling, logging configuration, and user-friendly error messages.
"""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime
import os
from pathlib import Path


class RAGErrorHandler:
    """Centralized error handler for the RAG application"""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = self._setup_logging(log_level)
        self.error_counts = {}
        
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup comprehensive logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Create logger
        logger = logging.getLogger("rag_application")
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler for all logs
        file_handler = logging.FileHandler(log_dir / "rag_application.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.FileHandler(log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        
        return logger
    
    def log_error(self, error: Exception, context: str = "", user_message: str = "") -> str:
        """
        Log an error with full context and return user-friendly message
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            user_message: Custom user-friendly message
            
        Returns:
            User-friendly error message
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Track error frequency
        error_key = f"{error_type}:{context}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log detailed error information
        self.logger.error(
            f"Error in {context}: {error_type}: {error_msg}\n"
            f"Traceback: {traceback.format_exc()}\n"
            f"Error count for this type: {self.error_counts[error_key]}"
        )
        
        # Return user-friendly message
        if user_message:
            return user_message
        
        return self._get_user_friendly_message(error, context)
    
    def _get_user_friendly_message(self, error: Exception, context: str) -> str:
        """Generate user-friendly error messages based on error type and context"""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # OpenAI API errors
        if "openai" in error_msg or "api" in error_msg:
            if "authentication" in error_msg or "api key" in error_msg:
                return "Authentication failed. Please check your OpenAI API key in the environment settings."
            elif "rate limit" in error_msg:
                return "API rate limit exceeded. Please wait a moment and try again."
            elif "quota" in error_msg:
                return "API quota exceeded. Please check your OpenAI account billing."
            else:
                return "OpenAI API error occurred. Please check your internet connection and API key."
        
        # Milvus/Vector store errors
        if "milvus" in error_msg or "vector" in error_msg:
            if "connection" in error_msg:
                return "Cannot connect to the vector database. Please ensure Milvus is running (docker-compose up -d)."
            elif "collection" in error_msg:
                return "Vector database collection error. Try reprocessing your documents."
            else:
                return "Vector database error occurred. Please check if the database service is running."
        
        # Document processing errors
        if context == "document_processing":
            if "pdf" in error_msg:
                return "Error processing PDF document. Please check if the file is valid and not corrupted."
            elif "docling" in error_msg:
                return "Document conversion failed. Please ensure the PDF is readable and try again."
            else:
                return "Document processing error. Please check your document files and try again."
        
        # Chunking strategy errors
        if context == "chunking":
            return "Error applying chunking strategy. Falling back to default settings."
        
        # Configuration errors
        if "config" in error_msg or context == "configuration":
            return "Configuration error. Please check your settings and try again."
        
        # File system errors
        if error_type in ["FileNotFoundError", "PermissionError", "OSError"]:
            if "permission" in error_msg:
                return "File permission error. Please check file permissions and try again."
            elif "not found" in error_msg:
                return "Required file not found. Please check your file paths and try again."
            else:
                return "File system error occurred. Please check your file paths and permissions."
        
        # Network errors
        if error_type in ["ConnectionError", "TimeoutError", "URLError"]:
            return "Network connection error. Please check your internet connection and try again."
        
        # Validation errors
        if error_type == "ValueError" and context == "validation":
            return f"Invalid input: {str(error)}"
        
        # Generic fallback
        return f"An unexpected error occurred. Please try again or contact support if the problem persists."
    
    def handle_startup_error(self, error: Exception, component: str) -> bool:
        """
        Handle startup errors with specific guidance
        
        Args:
            error: The startup error
            component: Component that failed to start
            
        Returns:
            True if the error is recoverable, False if fatal
        """
        error_msg = self.log_error(error, f"startup_{component}")
        
        print(f"❌ Failed to initialize {component}: {error_msg}")
        
        # Provide specific guidance based on component
        if component == "openai":
            print("   💡 Solution: Set your OpenAI API key in the .env file")
            print("      OPENAI_API_KEY=sk-your-actual-key-here")
            return False
        
        elif component == "milvus":
            print("   💡 Solution: Start the Milvus service")
            print("      docker-compose up -d")
            return False
        
        elif component == "document_processor":
            print("   💡 Solution: Check if the data directory exists and contains PDF files")
            print("      mkdir -p data && ls data/")
            return True  # Can continue without documents initially
        
        elif component == "vector_store":
            print("   💡 Solution: Ensure Milvus is running and accessible")
            print("      docker-compose ps")
            return False
        
        return True
    
    def create_error_context(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Create error context for better debugging"""
        return {
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "context": kwargs
        }


# Global error handler instance
error_handler = RAGErrorHandler(log_level=os.getenv("LOG_LEVEL", "INFO"))


def handle_errors(operation: str = "", user_message: str = ""):
    """
    Decorator for handling errors in functions
    
    Args:
        operation: Description of the operation being performed
        user_message: Custom user-friendly error message
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = operation or func.__name__
                error_msg = error_handler.log_error(e, context, user_message)
                
                # For API endpoints, return error response
                if hasattr(func, '__name__') and 'route' in str(func):
                    from flask import jsonify
                    return jsonify({'error': error_msg}), 500
                
                # For other functions, re-raise with context
                raise RuntimeError(error_msg) from e
        
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default_return=None, context: str = "", **kwargs):
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Value to return if function fails
        context: Context for error logging
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return if error occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_handler.log_error(e, context or func.__name__)
        return default_return


class ErrorRecovery:
    """Handles error recovery strategies"""
    
    @staticmethod
    def retry_with_backoff(func: Callable, max_retries: int = 3, backoff_factor: float = 1.0):
        """Retry function with exponential backoff"""
        import time
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                wait_time = backoff_factor * (2 ** attempt)
                error_handler.logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}"
                )
                time.sleep(wait_time)
    
    @staticmethod
    def fallback_chain(*functions):
        """Try functions in order until one succeeds"""
        last_error = None
        
        for func in functions:
            try:
                return func()
            except Exception as e:
                last_error = e
                error_handler.logger.warning(f"Fallback function failed: {str(e)}")
                continue
        
        if last_error:
            raise last_error
        
        raise RuntimeError("All fallback functions failed")


# Validation helpers
class InputValidator:
    """Input validation and sanitization"""
    
    @staticmethod
    def validate_question(question: str) -> str:
        """Validate and sanitize user question input"""
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
        
        # Strip whitespace
        question = question.strip()
        
        if not question:
            raise ValueError("Question cannot be empty")
        
        # Length validation
        if len(question) > 5000:
            raise ValueError("Question is too long (maximum 5000 characters)")
        
        if len(question) < 3:
            raise ValueError("Question is too short (minimum 3 characters)")
        
        # Basic sanitization - remove potentially harmful characters
        import re
        question = re.sub(r'[<>"\']', '', question)
        
        return question
    
    @staticmethod
    def validate_chunking_config(strategy: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate chunking configuration"""
        if not strategy or not isinstance(strategy, str):
            raise ValueError("Strategy must be a non-empty string")
        
        valid_strategies = ['recursive', 'semantic', 'token', 'paragraph', 'hybrid']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {', '.join(valid_strategies)}")
        
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Validate numeric parameters
        numeric_fields = {
            'chunk_size': (100, 10000),
            'chunk_overlap': (0, 1000),
            'similarity_threshold': (0.1, 1.0),
            'max_tokens': (100, 8000),
            'min_chunk_size': (50, 2000),
            'max_chunk_size': (500, 20000)
        }
        
        validated_config = {}
        for key, value in config.items():
            if key in numeric_fields:
                try:
                    num_value = float(value)
                    min_val, max_val = numeric_fields[key]
                    if not (min_val <= num_value <= max_val):
                        raise ValueError(f"{key} must be between {min_val} and {max_val}")
                    validated_config[key] = int(num_value) if key != 'similarity_threshold' else num_value
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid value for {key}: must be a number")
            else:
                validated_config[key] = value
        
        return validated_config
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        import re
        
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        # Remove path separators and dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = filename.replace('..', '')
        
        if not filename:
            raise ValueError("Invalid filename")
        
        return filename


# Export main components
__all__ = [
    'error_handler',
    'handle_errors',
    'safe_execute',
    'ErrorRecovery',
    'InputValidator',
    'RAGErrorHandler'
]