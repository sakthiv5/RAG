"""
Environment variable management and validation for the RAG application.
"""
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnvVariable:
    """Configuration for an environment variable"""
    name: str
    description: str
    required: bool = True
    default: Optional[str] = None
    validation_func: Optional[callable] = None
    sensitive: bool = False  # Whether to mask the value in logs


class EnvironmentManager:
    """Manages environment variables and validation"""
    
    def __init__(self):
        self.env_vars = self._define_environment_variables()
        self.validation_errors = []
        self.warnings = []

    def _define_environment_variables(self) -> Dict[str, EnvVariable]:
        """Define all required and optional environment variables"""
        return {
            'OPENAI_API_KEY': EnvVariable(
                name='OPENAI_API_KEY',
                description='OpenAI API key for GPT-4-mini and embeddings',
                required=True,
                validation_func=self._validate_openai_key,
                sensitive=True
            ),
            'MINIO_USERNAME': EnvVariable(
                name='MINIO_USERNAME',
                description='MinIO username for Milvus object storage',
                required=False,
                default='minioadmin'
            ),
            'MINIO_PASSWORD': EnvVariable(
                name='MINIO_PASSWORD',
                description='MinIO password for Milvus object storage',
                required=False,
                default='minioadmin',
                sensitive=True
            ),
            'MILVUS_HOST': EnvVariable(
                name='MILVUS_HOST',
                description='Milvus database host',
                required=False,
                default='localhost'
            ),
            'MILVUS_PORT': EnvVariable(
                name='MILVUS_PORT',
                description='Milvus database port',
                required=False,
                default='19530',
                validation_func=self._validate_port
            ),
            'FLASK_SECRET_KEY': EnvVariable(
                name='FLASK_SECRET_KEY',
                description='Flask application secret key',
                required=False,
                default='dev-secret-key',
                sensitive=True
            ),
            'FLASK_DEBUG': EnvVariable(
                name='FLASK_DEBUG',
                description='Flask debug mode (true/false)',
                required=False,
                default='false',
                validation_func=self._validate_boolean
            ),
            'LOG_LEVEL': EnvVariable(
                name='LOG_LEVEL',
                description='Application log level (DEBUG, INFO, WARNING, ERROR)',
                required=False,
                default='INFO',
                validation_func=self._validate_log_level
            ),
            'DATA_DIR': EnvVariable(
                name='DATA_DIR',
                description='Directory containing PDF documents to process',
                required=False,
                default='data',
                validation_func=self._validate_directory
            ),
            'CONFIG_DIR': EnvVariable(
                name='CONFIG_DIR',
                description='Directory for configuration files',
                required=False,
                default='config'
            ),
            'FLASK_HOST': EnvVariable(
                name='FLASK_HOST',
                description='Flask application host address',
                required=False,
                default='0.0.0.0'
            ),
            'FLASK_PORT': EnvVariable(
                name='FLASK_PORT',
                description='Flask application port',
                required=False,
                default='5000',
                validation_func=self._validate_port
            )
        }

    def _validate_openai_key(self, value: str) -> bool:
        """Validate OpenAI API key format"""
        if not value or value == 'your-new-key-here':
            return False
        # OpenAI keys typically start with 'sk-'
        if not value.startswith('sk-'):
            return False
        # Should be at least 40 characters
        if len(value) < 40:
            return False
        return True

    def _validate_port(self, value: str) -> bool:
        """Validate port number"""
        try:
            port = int(value)
            return 1 <= port <= 65535
        except ValueError:
            return False

    def _validate_boolean(self, value: str) -> bool:
        """Validate boolean value"""
        return value.lower() in ['true', 'false', '1', '0', 'yes', 'no']

    def _validate_log_level(self, value: str) -> bool:
        """Validate log level"""
        return value.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

    def _validate_directory(self, value: str) -> bool:
        """Validate directory exists or can be created"""
        try:
            path = Path(value)
            if path.exists():
                return path.is_dir()
            else:
                # Try to create the directory
                path.mkdir(parents=True, exist_ok=True)
                return True
        except Exception:
            return False

    def validate_environment(self) -> bool:
        """Validate all environment variables"""
        self.validation_errors = []
        self.warnings = []
        
        for var_name, env_var in self.env_vars.items():
            value = os.getenv(var_name)
            
            # Check if required variable is missing
            if env_var.required and not value:
                self.validation_errors.append(
                    f"Missing required environment variable: {var_name}\n"
                    f"  Description: {env_var.description}\n"
                    f"  Please set this variable in your .env file or environment"
                )
                continue
            
            # Use default if not set and not required
            if not value and env_var.default is not None:
                os.environ[var_name] = env_var.default
                value = env_var.default
                self.warnings.append(
                    f"Using default value for {var_name}: "
                    f"{'***' if env_var.sensitive else env_var.default}"
                )
            
            # Validate value if validation function exists
            if value and env_var.validation_func:
                if not env_var.validation_func(value):
                    self.validation_errors.append(
                        f"Invalid value for environment variable: {var_name}\n"
                        f"  Current value: {'***' if env_var.sensitive else value}\n"
                        f"  Description: {env_var.description}\n"
                        f"  Please check the format and try again"
                    )
        
        return len(self.validation_errors) == 0

    def get_validation_report(self) -> Dict[str, Any]:
        """Get detailed validation report"""
        return {
            'valid': len(self.validation_errors) == 0,
            'errors': self.validation_errors,
            'warnings': self.warnings,
            'environment_status': self._get_environment_status()
        }

    def _get_environment_status(self) -> Dict[str, Any]:
        """Get status of all environment variables"""
        status = {}
        for var_name, env_var in self.env_vars.items():
            value = os.getenv(var_name)
            status[var_name] = {
                'set': value is not None,
                'value': '***' if env_var.sensitive and value else value,
                'using_default': value == env_var.default,
                'required': env_var.required,
                'description': env_var.description
            }
        return status

    def print_validation_errors(self):
        """Print validation errors in a user-friendly format"""
        if not self.validation_errors:
            print("✅ All environment variables are valid!")
            return
        
        print("❌ Environment Variable Validation Errors:")
        print("=" * 50)
        
        for i, error in enumerate(self.validation_errors, 1):
            print(f"\n{i}. {error}")
        
        print("\n" + "=" * 50)
        print("Please fix these issues and restart the application.")
        
        # Print helpful setup instructions
        self._print_setup_instructions()

    def print_warnings(self):
        """Print validation warnings"""
        if not self.warnings:
            return
        
        print("\n⚠️  Environment Variable Warnings:")
        print("-" * 40)
        
        for warning in self.warnings:
            print(f"  • {warning}")
        
        print()

    def _print_setup_instructions(self):
        """Print setup instructions for missing variables"""
        print("\n📋 Setup Instructions:")
        print("-" * 30)
        print("1. Create or update your .env file in the project root")
        print("2. Add the missing environment variables:")
        print()
        
        for error in self.validation_errors:
            if "OPENAI_API_KEY" in error:
                print("   OPENAI_API_KEY=sk-your-actual-openai-api-key-here")
                print("   (Get your API key from: https://platform.openai.com/api-keys)")
                print()
        
        print("3. Restart the application")
        print("\nExample .env file:")
        print("-" * 20)
        print("OPENAI_API_KEY=sk-your-actual-key-here")
        print("MINIO_USERNAME=minioadmin")
        print("MINIO_PASSWORD=minioadmin")
        print("FLASK_DEBUG=false")
        print("LOG_LEVEL=INFO")

    def create_env_template(self, template_path: str = ".env.template") -> bool:
        """Create a template .env file with all variables"""
        try:
            template_content = []
            template_content.append("# Environment Configuration for RAG Application")
            template_content.append("# Copy this file to .env and fill in your actual values")
            template_content.append("")
            
            for var_name, env_var in self.env_vars.items():
                template_content.append(f"# {env_var.description}")
                if env_var.required:
                    template_content.append(f"# REQUIRED")
                else:
                    template_content.append(f"# Optional (default: {env_var.default})")
                
                if env_var.sensitive:
                    placeholder = "your-secret-here"
                elif var_name == "OPENAI_API_KEY":
                    placeholder = "sk-your-openai-api-key-here"
                else:
                    placeholder = env_var.default or "value-here"
                
                template_content.append(f"{var_name}={placeholder}")
                template_content.append("")
            
            with open(template_path, 'w') as f:
                f.write('\n'.join(template_content))
            
            print(f"Environment template created: {template_path}")
            return True
            
        except Exception as e:
            print(f"Error creating environment template: {e}")
            return False

    def check_critical_services(self) -> Dict[str, bool]:
        """Check if critical external services are accessible"""
        services_status = {}
        
        # Check OpenAI API
        try:
            import openai
            openai.api_key = os.getenv('OPENAI_API_KEY')
            # This is a simple check - in production you might want to make a test API call
            services_status['openai'] = bool(openai.api_key and self._validate_openai_key(openai.api_key))
        except ImportError:
            services_status['openai'] = False
        except Exception:
            services_status['openai'] = False
        
        # Check Milvus connection (basic port check)
        try:
            import socket
            host = os.getenv('MILVUS_HOST', 'localhost')
            port = int(os.getenv('MILVUS_PORT', '19530'))
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            services_status['milvus'] = result == 0
            sock.close()
        except Exception:
            services_status['milvus'] = False
        
        return services_status


# Global environment manager instance
env_manager = EnvironmentManager()


def validate_startup_environment() -> bool:
    """Validate environment on application startup"""
    print("🔍 Validating environment configuration...")
    
    is_valid = env_manager.validate_environment()
    
    if not is_valid:
        env_manager.print_validation_errors()
        return False
    
    env_manager.print_warnings()
    
    # Check critical services
    print("🔗 Checking external service connectivity...")
    services = env_manager.check_critical_services()
    
    for service, status in services.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {service.title()}: {'Connected' if status else 'Not accessible'}")
    
    if not all(services.values()):
        print("\n⚠️  Some services are not accessible. The application may not work correctly.")
        print("   Please ensure Docker services are running: docker-compose up -d")
    
    print("✅ Environment validation complete!")
    return True


if __name__ == "__main__":
    # Test the environment validation
    validate_startup_environment()