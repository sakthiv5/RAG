"""
Document processing service for RAG system.
Handles PDF conversion and chunking with various strategies.
"""
import os
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
from langchain.schema import Document
from chunking_strategies import ChunkingStrategyFactory
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles PDF conversion and document chunking"""
    
    def __init__(self, pdf_directory: str = "data"):
        self.pdf_directory = pdf_directory
        self.converter = DocumentConverter()
        
    def convert_pdfs(self) -> List[Document]:
        """
        Convert all PDFs in the directory to Document objects.
        
        Returns:
            List of Document objects with content and metadata
        """
        documents = []
        
        if not os.path.exists(self.pdf_directory):
            logger.warning(f"PDF directory {self.pdf_directory} does not exist")
            return documents
        
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith(".pdf")]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_directory}")
            return documents
        
        for filename in pdf_files:
            try:
                filepath = os.path.join(self.pdf_directory, filename)
                logger.info(f"Converting {filename}")
                
                result = self.converter.convert(filepath)
                doc = result.document
                markdown_text = doc.export_to_markdown()
                
                document = Document(
                    page_content=markdown_text,
                    metadata={
                        "source": filename,
                        "filepath": filepath,
                        "document_type": "pdf",
                        "original_size": len(markdown_text)
                    }
                )
                documents.append(document)
                
            except Exception as e:
                logger.error(f"Error converting {filename}: {str(e)}")
                continue
        
        logger.info(f"Successfully converted {len(documents)} PDF files")
        return documents
    
    def chunk_documents(self, documents: List[Document], strategy: str, config: Dict[str, Any]) -> List[Document]:
        """
        Chunk documents using the specified strategy.
        
        Args:
            documents: List of documents to chunk
            strategy: Name of the chunking strategy
            config: Configuration for the chunking strategy
            
        Returns:
            List of chunked documents
        """
        try:
            chunking_strategy = self.get_strategy_instance(strategy, config)
            chunks = chunking_strategy.chunk_documents(documents)
            
            logger.info(f"Chunked {len(documents)} documents into {len(chunks)} chunks using {strategy} strategy")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking documents with {strategy} strategy: {str(e)}")
            # Fallback to recursive strategy
            logger.info("Falling back to recursive chunking strategy")
            fallback_config = {
                'chunk_size': 1000,
                'chunk_overlap': 100,
                'separators': ["\n##", "\n#", "\n- ", "\n", " "]
            }
            fallback_strategy = self.get_strategy_instance('recursive', fallback_config)
            return fallback_strategy.chunk_documents(documents)
    
    def get_strategy_instance(self, strategy: str, config: Dict[str, Any]):
        """
        Get an instance of the specified chunking strategy.
        
        Args:
            strategy: Name of the chunking strategy
            config: Configuration for the strategy
            
        Returns:
            ChunkingStrategy instance
        """
        return ChunkingStrategyFactory.create_strategy(strategy, config)
    
    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available chunking strategies with their default configurations.
        
        Returns:
            Dictionary of strategy names and their configurations
        """
        strategies = {
            'recursive': {
                'name': 'Recursive Character Text Splitter',
                'description': 'Splits text recursively using character separators',
                'config': {
                    'chunk_size': 1000,
                    'chunk_overlap': 100,
                    'separators': ["\n##", "\n#", "\n- ", "\n", " "]
                },
                'config_fields': [
                    {'name': 'chunk_size', 'type': 'number', 'label': 'Chunk Size', 'min': 100, 'max': 5000},
                    {'name': 'chunk_overlap', 'type': 'number', 'label': 'Chunk Overlap', 'min': 0, 'max': 500},
                    {'name': 'separators', 'type': 'text', 'label': 'Separators (comma-separated)', 'value': '\\n##,\\n#,\\n- ,\\n, '}
                ]
            },
            'semantic': {
                'name': 'Semantic Chunking',
                'description': 'Groups sentences by semantic similarity',
                'config': {
                    'similarity_threshold': 0.8,
                    'min_chunk_size': 200
                },
                'config_fields': [
                    {'name': 'similarity_threshold', 'type': 'number', 'label': 'Similarity Threshold', 'min': 0.1, 'max': 1.0, 'step': 0.1},
                    {'name': 'min_chunk_size', 'type': 'number', 'label': 'Minimum Chunk Size', 'min': 50, 'max': 1000}
                ]
            },
            'token': {
                'name': 'Token-based Chunking',
                'description': 'Splits text based on token count limits',
                'config': {
                    'max_tokens': 1000,
                    'tokenizer_model': 'gpt-4'
                },
                'config_fields': [
                    {'name': 'max_tokens', 'type': 'number', 'label': 'Max Tokens per Chunk', 'min': 100, 'max': 4000},
                    {'name': 'tokenizer_model', 'type': 'select', 'label': 'Tokenizer Model', 'options': ['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003']}
                ]
            },
            'paragraph': {
                'name': 'Paragraph-based Chunking',
                'description': 'Splits text at paragraph boundaries',
                'config': {
                    'min_chunk_size': 200,
                    'max_chunk_size': 2000
                },
                'config_fields': [
                    {'name': 'min_chunk_size', 'type': 'number', 'label': 'Minimum Chunk Size', 'min': 50, 'max': 1000},
                    {'name': 'max_chunk_size', 'type': 'number', 'label': 'Maximum Chunk Size', 'min': 500, 'max': 5000}
                ]
            },
            'hybrid': {
                'name': 'Hybrid Chunking',
                'description': 'Combines multiple chunking strategies',
                'config': {
                    'chunk_size': 1000,
                    'similarity_threshold': 0.8,
                    'max_tokens': 1000
                },
                'config_fields': [
                    {'name': 'chunk_size', 'type': 'number', 'label': 'Base Chunk Size', 'min': 100, 'max': 5000},
                    {'name': 'similarity_threshold', 'type': 'number', 'label': 'Similarity Threshold', 'min': 0.1, 'max': 1.0, 'step': 0.1},
                    {'name': 'max_tokens', 'type': 'number', 'label': 'Max Tokens', 'min': 100, 'max': 4000}
                ]
            }
        }
        
        return strategies
    
    def validate_config(self, strategy: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize configuration for a chunking strategy.
        
        Args:
            strategy: Name of the chunking strategy
            config: Configuration to validate
            
        Returns:
            Validated and sanitized configuration
        """
        strategies = self.get_available_strategies()
        
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        default_config = strategies[strategy]['config']
        validated_config = default_config.copy()
        
        # Update with provided config, ensuring types are correct
        for key, value in config.items():
            if key in default_config:
                try:
                    # Type conversion based on default value type
                    if isinstance(default_config[key], int):
                        validated_config[key] = int(value)
                    elif isinstance(default_config[key], float):
                        validated_config[key] = float(value)
                    elif isinstance(default_config[key], list):
                        if isinstance(value, str):
                            # Handle comma-separated strings for separators
                            validated_config[key] = [s.strip() for s in value.split(',')]
                        else:
                            validated_config[key] = value
                    else:
                        validated_config[key] = value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {key}: {value}. Using default: {default_config[key]}")
        
        return validated_config