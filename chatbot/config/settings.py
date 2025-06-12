import os
from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from typing import Optional

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Kibwa Chatbot"
    DEBUG: bool = True
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    
    # AWS
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_DEFAULT_REGION: str = "ap-southeast-2"
    AWS_S3_BUCKET: str = "kibwa-12"
    AWS_S3_PREFIX: str = "project/"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/sql_app.db"
    
    # Security
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    ALGORITHM: str = "HS256"
    
    # File paths
    STATIC_DIR: Path = BASE_DIR / "static"
    TEMPLATES_DIR: Path = BASE_DIR / "templates"
    LOGS_DIR: Path = BASE_DIR / "logs"
    MEMBER_DIR: Path = BASE_DIR / "member_information"
    CHAT_LOGS_DIR: Path = BASE_DIR / "chat_logs"
    
    # Create necessary directories
    @validator('STATIC_DIR', 'TEMPLATES_DIR', 'LOGS_DIR', 'MEMBER_DIR', 'CHAT_LOGS_DIR', pre=True)
    def create_directories(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

# Initialize settings
settings = Settings()

# Ensure all directories exist
for directory in [settings.STATIC_DIR, settings.TEMPLATES_DIR, settings.LOGS_DIR, 
                 settings.MEMBER_DIR, settings.CHAT_LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
