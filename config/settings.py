"""Application configuration settings"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings and configuration"""
    
    # API Keys
    DEEPL_API_KEY: Optional[str] = os.getenv('DEEPL_API_KEY')
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    AZURE_TRANSLATOR_KEY: Optional[str] = os.getenv('AZURE_TRANSLATOR_KEY')
    AZURE_TRANSLATOR_REGION: Optional[str] = os.getenv('AZURE_TRANSLATOR_REGION', 'global')
    
    # DeepLX Configuration
    DEEPLX_ENDPOINT: Optional[str] = os.getenv('DEEPLX_ENDPOINT')
    
    # File handling
    MAX_FILE_SIZE_MB: int = int(os.getenv('MAX_FILE_SIZE_MB', '10'))
    ALLOWED_EXTENSIONS: list = os.getenv('ALLOWED_EXTENSIONS', '.pdf,.docx,.txt').split(',')
    UPLOAD_DIR: str = os.getenv('UPLOAD_DIR', 'uploads')
    DOWNLOAD_DIR: str = os.getenv('DOWNLOAD_DIR', 'downloads')
    TEMP_DIR: str = os.getenv('TEMP_DIR', 'temp')
    
    # Translation settings
    DEFAULT_BATCH_SIZE: int = 50
    MAX_TEXT_LENGTH: int = 5000
    TRANSLATION_TIMEOUT: int = 300  # 5 minutes
    
    # Server settings
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', 8000))
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Security
    SECRET_KEY: str = os.getenv('SECRET_KEY')
    if not SECRET_KEY:
        raise RuntimeError('SECRET_KEY environment variable not set. Set SECRET_KEY before starting the application.')
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', 'app.log')
    
    # Cleanup settings
    CLEANUP_INTERVAL_HOURS: int = 24
    MAX_FILE_AGE_HOURS: int = 48
    
    def __init__(self):
        # Create required directories
        for directory in [self.UPLOAD_DIR, self.DOWNLOAD_DIR, self.TEMP_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def get_available_translators(self) -> list:
        """Get list of available translation services"""
        available = []
        
        if self.DEEPL_API_KEY:
            available.append('deepl')
        
        if self.GOOGLE_APPLICATION_CREDENTIALS:
            available.append('google')
        
        if self.AZURE_TRANSLATOR_KEY:
            available.append('azure')
        
        if self.DEEPLX_ENDPOINT:
            available.append('deeplx')

        return available
        
    def validate_config(self) -> dict:
        """Validate configuration and return status"""
        issues = []
        
        # Check translation services
        available_translators = self.get_available_translators()
        if not available_translators:
            issues.append("No translation services configured")
        
        # Validate API key formats
        if self.DEEPL_API_KEY and not self.DEEPL_API_KEY.endswith(':fx'):
            issues.append("DeepL API key format may be invalid")
        
        # Check SECRET_KEY presence
        if not self.SECRET_KEY:
            issues.append("SECRET_KEY environment variable is missing")
        
        # Check directory permissions
        for directory in [self.UPLOAD_DIR, self.DOWNLOAD_DIR, self.TEMP_DIR]:
            if os.path.exists(directory) and not os.access(directory, os.W_OK):
                issues.append(f"Directory not writable: {directory}")
        
        # Check directories
        for directory in [self.UPLOAD_DIR, self.DOWNLOAD_DIR, self.TEMP_DIR]:
            if not os.path.exists(directory):
                issues.append(f"Directory does not exist: {directory}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "available_translators": available_translators
        }