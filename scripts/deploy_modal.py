#!/usr/bin/env python3
"""
Modal Labs deployment script for the Dolphin OCR Translation project.

This script handles:
1. Deploying the Dolphin OCR microservice to Modal
2. Setting up the main translation application
3. Configuring environment variables and secrets
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import modal


def check_environment_variables():
    """Check that all required environment variables are set."""
    required_vars = [
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET", 
        "LINGO_API_KEY",
    ]
    
    optional_vars = [
        "HF_TOKEN",  # HuggingFace token for model downloads
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    print("✅ All required environment variables are set")
    
    # Check optional variables
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ Optional variable {var} is set")
        else:
            print(f"⚠️  Optional variable {var} is not set")
    
    return True


def create_modal_secrets():
    """Create Modal secrets for the application."""
    print("🔐 Creating Modal secrets...")
    
    # Create translation API secret
    translation_secret_data = {
        "LINGO_API_KEY": os.getenv("LINGO_API_KEY"),
    }
    
    # Add HuggingFace token if available
    if os.getenv("HF_TOKEN"):
        translation_secret_data["HF_TOKEN"] = os.getenv("HF_TOKEN")
    
    try:
        # Create the secret using Modal CLI (this would need to be run separately)
        print("📝 Translation API secret data prepared")
        print("   Run this command to create the secret:")
        secret_cmd = "modal secret create translation-api"
        for key, value in translation_secret_data.items():
            secret_cmd += f" {key}={value}"
        print(f"   {secret_cmd}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create secrets: {e}")
        return False


def deploy_dolphin_service():
    """Deploy the Dolphin OCR service to Modal."""
    print("🚀 Deploying Dolphin OCR service...")
    
    try:
        # Import the Modal app
        from services.dolphin_modal_service import app as dolphin_app
        
        print("📦 Dolphin service app loaded successfully")
        print("   To deploy, run: modal deploy services/dolphin_modal_service.py")
        
        return True
    except Exception as e:
        print(f"❌ Failed to deploy Dolphin service: {e}")
        return False


def deploy_main_application():
    """Deploy the main translation application."""
    print("🚀 Deploying main translation application...")
    
    # This would be implemented when we create the main Modal app
    print("📝 Main application deployment not yet implemented")
    print("   This will include the Gradio UI and translation pipeline")
    
    return True


def main():
    """Main deployment function."""
    print("🌊 Modal Labs Deployment for Dolphin OCR Translation")
    print("=" * 60)
    
    # Step 1: Check environment variables
    if not check_environment_variables():
        sys.exit(1)
    
    # Step 2: Create Modal secrets
    if not create_modal_secrets():
        sys.exit(1)
    
    # Step 3: Deploy Dolphin service
    if not deploy_dolphin_service():
        sys.exit(1)
    
    # Step 4: Deploy main application
    if not deploy_main_application():
        sys.exit(1)
    
    print("\n✅ Deployment preparation complete!")
    print("\nNext steps:")
    print("1. Create the Modal secret:")
    print("   modal secret create translation-api LINGO_API_KEY=your-key")
    if os.getenv("HF_TOKEN"):
        print("   (HF_TOKEN will be included automatically)")
    
    print("\n2. Deploy the Dolphin OCR service:")
    print("   modal deploy services/dolphin_modal_service.py")
    
    print("\n3. Update DOLPHIN_ENDPOINT in your .env:")
    print("   DOLPHIN_ENDPOINT=https://modal-labs--dolphin-ocr-service-dolphin-ocr-endpoint.modal.run")
    
    print("\n4. Test the deployment:")
    print("   python -m scripts.test_modal_deployment")


if __name__ == "__main__":
    main()
