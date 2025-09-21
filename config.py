"""
Configuration management for chunking strategies and application settings.
"""
import os
import json
import shutil
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

@dataclass
class ChunkingConfig:
    """Base configuration for chunking strategies"""
    strategy: str
    chunk_size: int = 1000
    chunk_overlap: int = 100
    
    # Strategy-specific parameters
    separators: Optional[List[str]] = None
    similarity_threshold: Optional[float] = None
    max_tokens: Optional[int] = None
    tokenizer_model: Optional[str] = None
    min_chunk_size: Optional[int] = None
    max_chunk_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkingConfig':
        """Create instance from dictionary"""
        return cls(**data)

class ConfigManager:
    """Manages application configuration and chunking strategies"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / "chunking_config.json"
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self._load_config()

    def _load_config(self):
        """Load configuration from file or create defaults"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.current_config = ChunkingConfig.from_dict(data)
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"Error loading config: {e}. Using defaults.")
                self.current_config = self._get_default_config()
        else:
            self.current_config = self._get_default_config()
            self.save_config()

    def _get_default_config(self) -> ChunkingConfig:
        """Get default chunking configuration"""
        return ChunkingConfig(
            strategy="recursive",
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n##", "\n#", "\n- ", "\n", " "]
        )

    def save_config(self, create_backup: bool = False):
        """Save current configuration to file"""
        try:
            # Create backup if requested and config file exists
            if create_backup and self.config_file.exists():
                self.create_backup("auto")
            
            with open(self.config_file, 'w') as f:
                json.dump(self.current_config.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update configuration with new values"""
        try:
            # Validate the new configuration
            validated_config = self._validate_config(new_config)
            self.current_config = ChunkingConfig.from_dict(validated_config)
            self.save_config(create_backup=True)  # Auto-create backup on updates
            return True
        except Exception as e:
            print(f"Error updating config: {e}")
            return False

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters"""
        strategy = config.get('strategy', 'recursive')
        
        # Base validation
        validated = {
            'strategy': strategy,
            'chunk_size': max(100, min(10000, config.get('chunk_size', 1000))),
            'chunk_overlap': max(0, min(500, config.get('chunk_overlap', 100)))
        }

        # Strategy-specific validation
        if strategy == 'recursive':
            validated['separators'] = config.get('separators', ["\n##", "\n#", "\n- ", "\n", " "])
        
        elif strategy == 'semantic':
            validated['similarity_threshold'] = max(0.1, min(1.0, config.get('similarity_threshold', 0.8)))
            validated['min_chunk_size'] = max(50, config.get('min_chunk_size', 200))
        
        elif strategy == 'token':
            validated['max_tokens'] = max(50, min(8000, config.get('max_tokens', 1000)))
            validated['tokenizer_model'] = config.get('tokenizer_model', 'gpt-4')
        
        elif strategy == 'paragraph':
            validated['min_chunk_size'] = max(50, config.get('min_chunk_size', 200))
            validated['max_chunk_size'] = max(500, config.get('max_chunk_size', 2000))
        
        elif strategy == 'hybrid':
            # Hybrid strategy can use multiple parameters
            validated.update({
                'similarity_threshold': config.get('similarity_threshold', 0.8),
                'max_tokens': config.get('max_tokens', 1000),
                'separators': config.get('separators', ["\n##", "\n#", "\n- ", "\n", " "])
            })

        return validated

    def get_strategy_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get available chunking strategies and their configurations"""
        return {
            'recursive': {
                'name': 'Recursive Character Text Splitter',
                'description': 'Splits text recursively using a hierarchy of separators',
                'parameters': {
                    'chunk_size': {
                        'type': 'number',
                        'label': 'Chunk Size',
                        'min': 100,
                        'max': 10000,
                        'default': 1000,
                        'description': 'Maximum size of each chunk in characters'
                    },
                    'chunk_overlap': {
                        'type': 'number',
                        'label': 'Chunk Overlap',
                        'min': 0,
                        'max': 500,
                        'default': 100,
                        'description': 'Number of overlapping characters between chunks'
                    },
                    'separators': {
                        'type': 'array',
                        'label': 'Separators',
                        'default': ["\n##", "\n#", "\n- ", "\n", " "],
                        'description': 'List of separators in order of preference'
                    }
                }
            },
            'semantic': {
                'name': 'Semantic Chunking',
                'description': 'Groups sentences based on semantic similarity',
                'parameters': {
                    'similarity_threshold': {
                        'type': 'number',
                        'label': 'Similarity Threshold',
                        'min': 0.1,
                        'max': 1.0,
                        'step': 0.1,
                        'default': 0.8,
                        'description': 'Minimum similarity score to group sentences'
                    },
                    'min_chunk_size': {
                        'type': 'number',
                        'label': 'Minimum Chunk Size',
                        'min': 50,
                        'max': 1000,
                        'default': 200,
                        'description': 'Minimum size of each chunk in characters'
                    }
                }
            },
            'token': {
                'name': 'Token-based Chunking',
                'description': 'Splits text based on token count using tiktoken',
                'parameters': {
                    'max_tokens': {
                        'type': 'number',
                        'label': 'Max Tokens per Chunk',
                        'min': 50,
                        'max': 8000,
                        'default': 1000,
                        'description': 'Maximum number of tokens per chunk'
                    },
                    'tokenizer_model': {
                        'type': 'select',
                        'label': 'Tokenizer Model',
                        'options': ['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003'],
                        'default': 'gpt-4',
                        'description': 'Model to use for tokenization'
                    }
                }
            },
            'paragraph': {
                'name': 'Paragraph-based Chunking',
                'description': 'Splits text on paragraph boundaries',
                'parameters': {
                    'min_chunk_size': {
                        'type': 'number',
                        'label': 'Minimum Chunk Size',
                        'min': 50,
                        'max': 1000,
                        'default': 200,
                        'description': 'Minimum size of each chunk in characters'
                    },
                    'max_chunk_size': {
                        'type': 'number',
                        'label': 'Maximum Chunk Size',
                        'min': 500,
                        'max': 5000,
                        'default': 2000,
                        'description': 'Maximum size of each chunk in characters'
                    }
                }
            },
            'hybrid': {
                'name': 'Hybrid Chunking',
                'description': 'Combines multiple chunking strategies',
                'parameters': {
                    'chunk_size': {
                        'type': 'number',
                        'label': 'Base Chunk Size',
                        'min': 100,
                        'max': 10000,
                        'default': 1000,
                        'description': 'Base chunk size for recursive splitting'
                    },
                    'similarity_threshold': {
                        'type': 'number',
                        'label': 'Similarity Threshold',
                        'min': 0.1,
                        'max': 1.0,
                        'step': 0.1,
                        'default': 0.8,
                        'description': 'Similarity threshold for semantic grouping'
                    },
                    'max_tokens': {
                        'type': 'number',
                        'label': 'Max Tokens',
                        'min': 50,
                        'max': 8000,
                        'default': 1000,
                        'description': 'Maximum tokens per final chunk'
                    }
                }
            }
        }

    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        return self.current_config.to_dict()

    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        # Create backup before resetting
        self.create_backup("before_reset")
        self.current_config = self._get_default_config()
        self.save_config()

    def create_backup(self, suffix: str = None) -> str:
        """Create a backup of the current configuration"""
        try:
            if not self.config_file.exists():
                raise FileNotFoundError("No configuration file to backup")
            
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if suffix:
                backup_filename = f"chunking_config_{timestamp}_{suffix}.json"
            else:
                backup_filename = f"chunking_config_{timestamp}.json"
            
            backup_path = self.backup_dir / backup_filename
            
            # Copy current config to backup
            shutil.copy2(self.config_file, backup_path)
            
            print(f"Configuration backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            print(f"Error creating backup: {e}")
            raise

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available configuration backups"""
        try:
            backups = []
            for backup_file in self.backup_dir.glob("chunking_config_*.json"):
                try:
                    # Extract timestamp from filename
                    filename = backup_file.stem
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        date_part = parts[2]
                        time_part = parts[3] if len(parts) > 3 else "000000"
                        
                        # Parse timestamp
                        timestamp_str = f"{date_part}_{time_part}"
                        timestamp = datetime.strptime(timestamp_str[:15], "%Y%m%d_%H%M%S")
                        
                        # Get file size
                        file_size = backup_file.stat().st_size
                        
                        # Try to load config to get strategy info
                        try:
                            with open(backup_file, 'r') as f:
                                config_data = json.load(f)
                                strategy = config_data.get('strategy', 'unknown')
                        except:
                            strategy = 'unknown'
                        
                        backups.append({
                            'filename': backup_file.name,
                            'path': str(backup_file),
                            'timestamp': timestamp.isoformat(),
                            'size': file_size,
                            'strategy': strategy,
                            'created': timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                except Exception as e:
                    print(f"Error processing backup file {backup_file}: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            backups.sort(key=lambda x: x['timestamp'], reverse=True)
            return backups
            
        except Exception as e:
            print(f"Error listing backups: {e}")
            return []

    def restore_from_backup(self, backup_filename: str) -> bool:
        """Restore configuration from a backup file"""
        try:
            backup_path = self.backup_dir / backup_filename
            
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_filename}")
            
            # Create backup of current config before restoring
            self.create_backup("before_restore")
            
            # Validate the backup file by trying to load it
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Validate the configuration
            validated_config = self._validate_config(backup_data)
            
            # Copy backup to current config
            shutil.copy2(backup_path, self.config_file)
            
            # Reload configuration
            self._load_config()
            
            print(f"Configuration restored from backup: {backup_filename}")
            return True
            
        except Exception as e:
            print(f"Error restoring from backup: {e}")
            return False

    def delete_backup(self, backup_filename: str) -> bool:
        """Delete a specific backup file"""
        try:
            backup_path = self.backup_dir / backup_filename
            
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_filename}")
            
            backup_path.unlink()
            print(f"Backup deleted: {backup_filename}")
            return True
            
        except Exception as e:
            print(f"Error deleting backup: {e}")
            return False

    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """Clean up old backup files, keeping only the most recent ones"""
        try:
            backups = self.list_backups()
            
            if len(backups) <= keep_count:
                return 0
            
            # Delete oldest backups
            deleted_count = 0
            for backup in backups[keep_count:]:
                if self.delete_backup(backup['filename']):
                    deleted_count += 1
            
            print(f"Cleaned up {deleted_count} old backup files")
            return deleted_count
            
        except Exception as e:
            print(f"Error cleaning up backups: {e}")
            return 0

    def export_config(self, export_path: str) -> bool:
        """Export current configuration to a specified path"""
        try:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata to export
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'version': '1.0',
                'config': self.current_config.to_dict()
            }
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Configuration exported to: {export_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting configuration: {e}")
            return False

    def import_config(self, import_path: str) -> bool:
        """Import configuration from a specified path"""
        try:
            import_file = Path(import_path)
            
            if not import_file.exists():
                raise FileNotFoundError(f"Import file not found: {import_path}")
            
            with open(import_file, 'r') as f:
                import_data = json.load(f)
            
            # Handle both direct config and exported format
            if 'config' in import_data:
                config_data = import_data['config']
            else:
                config_data = import_data
            
            # Create backup before importing
            self.create_backup("before_import")
            
            # Validate and update configuration
            if self.update_config(config_data):
                print(f"Configuration imported from: {import_path}")
                return True
            else:
                print("Failed to import configuration: validation failed")
                return False
            
        except Exception as e:
            print(f"Error importing configuration: {e}")
            return False

# Global configuration manager instance
config_manager = ConfigManager()