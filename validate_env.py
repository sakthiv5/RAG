#!/usr/bin/env python3
"""
Standalone environment validation script for the RAG application.
Run this script to check your environment configuration before starting the application.
"""

import sys
from dotenv import load_dotenv
from env_manager import env_manager

def main():
    """Main validation function"""
    print("🔍 RAG Application Environment Validator")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Validate environment
    is_valid = env_manager.validate_environment()
    
    # Print results
    if is_valid:
        print("✅ Environment validation passed!")
        env_manager.print_warnings()
        
        # Check services
        print("\n🔗 Checking external services...")
        services = env_manager.check_critical_services()
        
        all_services_ok = True
        for service, status in services.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {service.title()}: {'Available' if status else 'Not accessible'}")
            if not status:
                all_services_ok = False
        
        if not all_services_ok:
            print("\n⚠️  Some services are not accessible.")
            print("   Make sure to start Docker services: docker-compose up -d")
            return 1
        
        print("\n🎉 All checks passed! You can start the application.")
        return 0
    else:
        env_manager.print_validation_errors()
        return 1

if __name__ == "__main__":
    sys.exit(main())