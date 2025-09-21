#!/usr/bin/env python3
"""
Configuration management CLI tool for the RAG application.
Provides commands to manage chunking configurations and backups.
"""

import sys
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from config import config_manager
from env_manager import env_manager

def list_backups():
    """List all configuration backups"""
    backups = config_manager.list_backups()
    
    if not backups:
        print("No configuration backups found.")
        return
    
    print("Configuration Backups:")
    print("-" * 60)
    print(f"{'Filename':<30} {'Created':<20} {'Strategy':<15} {'Size'}")
    print("-" * 60)
    
    for backup in backups:
        size_kb = backup['size'] / 1024
        print(f"{backup['filename']:<30} {backup['created']:<20} {backup['strategy']:<15} {size_kb:.1f}KB")

def create_backup(suffix=None):
    """Create a configuration backup"""
    try:
        backup_path = config_manager.create_backup(suffix)
        print(f"✅ Backup created: {backup_path}")
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return 1
    return 0

def restore_backup(filename):
    """Restore configuration from backup"""
    try:
        if config_manager.restore_from_backup(filename):
            print(f"✅ Configuration restored from: {filename}")
            return 0
        else:
            print(f"❌ Failed to restore from: {filename}")
            return 1
    except Exception as e:
        print(f"❌ Error restoring backup: {e}")
        return 1

def show_current_config():
    """Show current configuration"""
    config = config_manager.get_current_config()
    strategies = config_manager.get_strategy_definitions()
    
    print("Current Configuration:")
    print("-" * 30)
    print(f"Strategy: {config['strategy']}")
    
    strategy_info = strategies.get(config['strategy'], {})
    if strategy_info:
        print(f"Name: {strategy_info.get('name', 'Unknown')}")
        print(f"Description: {strategy_info.get('description', 'No description')}")
    
    print("\nParameters:")
    for key, value in config.items():
        if key != 'strategy' and value is not None:
            print(f"  {key}: {value}")

def list_strategies():
    """List available chunking strategies"""
    strategies = config_manager.get_strategy_definitions()
    
    print("Available Chunking Strategies:")
    print("=" * 50)
    
    for strategy_id, strategy_info in strategies.items():
        print(f"\n{strategy_id}:")
        print(f"  Name: {strategy_info['name']}")
        print(f"  Description: {strategy_info['description']}")
        print("  Parameters:")
        
        for param_name, param_info in strategy_info['parameters'].items():
            print(f"    - {param_name}: {param_info['description']}")

def export_config(path):
    """Export configuration to file"""
    try:
        if config_manager.export_config(path):
            print(f"✅ Configuration exported to: {path}")
            return 0
        else:
            print(f"❌ Failed to export configuration to: {path}")
            return 1
    except Exception as e:
        print(f"❌ Error exporting configuration: {e}")
        return 1

def import_config(path):
    """Import configuration from file"""
    try:
        if config_manager.import_config(path):
            print(f"✅ Configuration imported from: {path}")
            return 0
        else:
            print(f"❌ Failed to import configuration from: {path}")
            return 1
    except Exception as e:
        print(f"❌ Error importing configuration: {e}")
        return 1

def reset_config():
    """Reset configuration to defaults"""
    try:
        config_manager.reset_to_defaults()
        print("✅ Configuration reset to defaults")
        return 0
    except Exception as e:
        print(f"❌ Error resetting configuration: {e}")
        return 1

def cleanup_backups(keep_count):
    """Clean up old backups"""
    try:
        deleted = config_manager.cleanup_old_backups(keep_count)
        print(f"✅ Cleaned up {deleted} old backup files")
        return 0
    except Exception as e:
        print(f"❌ Error cleaning up backups: {e}")
        return 1

def validate_environment():
    """Validate environment configuration"""
    load_dotenv()
    
    is_valid = env_manager.validate_environment()
    
    if is_valid:
        print("✅ Environment validation passed!")
        env_manager.print_warnings()
        return 0
    else:
        env_manager.print_validation_errors()
        return 1

def create_env_template():
    """Create environment template file"""
    try:
        if env_manager.create_env_template():
            print("✅ Environment template created: .env.template")
            return 0
        else:
            print("❌ Failed to create environment template")
            return 1
    except Exception as e:
        print(f"❌ Error creating template: {e}")
        return 1

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="RAG Application Configuration Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config show                    # Show current configuration
  %(prog)s config list-strategies         # List available strategies
  %(prog)s config reset                   # Reset to defaults
  
  %(prog)s backup list                    # List all backups
  %(prog)s backup create manual           # Create backup with suffix
  %(prog)s backup restore backup.json     # Restore from backup
  %(prog)s backup cleanup 5               # Keep only 5 most recent backups
  
  %(prog)s export config.json             # Export configuration
  %(prog)s import config.json             # Import configuration
  
  %(prog)s env validate                   # Validate environment
  %(prog)s env template                   # Create .env template
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Config commands
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_action')
    
    config_subparsers.add_parser('show', help='Show current configuration')
    config_subparsers.add_parser('list-strategies', help='List available strategies')
    config_subparsers.add_parser('reset', help='Reset to default configuration')
    
    # Backup commands
    backup_parser = subparsers.add_parser('backup', help='Backup management')
    backup_subparsers = backup_parser.add_subparsers(dest='backup_action')
    
    backup_subparsers.add_parser('list', help='List all backups')
    
    create_parser = backup_subparsers.add_parser('create', help='Create backup')
    create_parser.add_argument('suffix', nargs='?', help='Optional backup suffix')
    
    restore_parser = backup_subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument('filename', help='Backup filename to restore')
    
    cleanup_parser = backup_subparsers.add_parser('cleanup', help='Clean up old backups')
    cleanup_parser.add_argument('keep', type=int, default=10, help='Number of backups to keep')
    
    # Import/Export commands
    export_parser = subparsers.add_parser('export', help='Export configuration')
    export_parser.add_argument('path', help='Export file path')
    
    import_parser = subparsers.add_parser('import', help='Import configuration')
    import_parser.add_argument('path', help='Import file path')
    
    # Environment commands
    env_parser = subparsers.add_parser('env', help='Environment management')
    env_subparsers = env_parser.add_subparsers(dest='env_action')
    
    env_subparsers.add_parser('validate', help='Validate environment configuration')
    env_subparsers.add_parser('template', help='Create .env template file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Load environment for config operations
    if args.command in ['config', 'backup', 'export', 'import']:
        load_dotenv()
    
    # Execute commands
    try:
        if args.command == 'config':
            if args.config_action == 'show':
                show_current_config()
                return 0
            elif args.config_action == 'list-strategies':
                list_strategies()
                return 0
            elif args.config_action == 'reset':
                return reset_config()
            else:
                config_parser.print_help()
                return 1
        
        elif args.command == 'backup':
            if args.backup_action == 'list':
                list_backups()
                return 0
            elif args.backup_action == 'create':
                return create_backup(args.suffix)
            elif args.backup_action == 'restore':
                return restore_backup(args.filename)
            elif args.backup_action == 'cleanup':
                return cleanup_backups(args.keep)
            else:
                backup_parser.print_help()
                return 1
        
        elif args.command == 'export':
            return export_config(args.path)
        
        elif args.command == 'import':
            return import_config(args.path)
        
        elif args.command == 'env':
            if args.env_action == 'validate':
                return validate_environment()
            elif args.env_action == 'template':
                return create_env_template()
            else:
                env_parser.print_help()
                return 1
        
        else:
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())