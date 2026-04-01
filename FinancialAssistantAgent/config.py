"""
Configuration Management
Centralized configuration for the Financial Assistant AI system
"""

import os
from pathlib import Path
from typing import Optional

# Get the root directory (where config.py is located)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    """Main configuration class"""
    
    # Model Configuration
    MODEL_PATH = os.path.join(ROOT_DIR, "models", "finance-Llama3-8B.Q2_K.gguf")
    MODEL_CONTEXT_SIZE = 512   # reduced from 2048 — smaller = faster prefill on CPU
    MODEL_N_THREADS = 8        # use more CPU threads (change to match your CPU core count)
    MODEL_N_BATCH = 128
    MODEL_TEMPERATURE = 0.7
    MODEL_TOP_P = 0.9
    MODEL_MAX_TOKENS = 60      # hard cap — ensures response in <60s at 1 tok/sec
    
    # RAG Configuration
    DOCS_FOLDER = os.path.join(ROOT_DIR, "docs")
    PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    RAG_CHUNK_SIZE = 1000
    RAG_CHUNK_OVERLAP = 200
    RAG_TOP_K = 3
    RAG_MIN_SCORE = 0.1
    
    # API Configuration
    BACKEND_HOST = "127.0.0.1"
    BACKEND_PORT = 8000
    BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
    FRONTEND_PORT = 7860
    REQUEST_TIMEOUT = 300
    
    # Database Configuration
    DATABASE_PATH = os.path.join(ROOT_DIR, "financial_assistant.db")
    
    # Logging Configuration
    LOGS_DIR = os.path.join(ROOT_DIR, "logs")
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # UI Configuration
    UI_THEME = "soft"
    UI_TITLE = "Financial Assistant AI"
    UI_DESCRIPTION = "Your intelligent financial advisor powered by AI and RAG technology"
    
    # Security Configuration
    CORS_ORIGINS = ["*"]  # In production, specify actual origins
    API_KEY_REQUIRED = False  # Set to True for production
    
    # Performance Configuration
    MAX_CONCURRENT_REQUESTS = 10
    RESPONSE_CACHE_SIZE = 100
    CACHE_TTL = 3600  # seconds
    
    # Feature Flags
    ENABLE_RAG = True
    ENABLE_ANALYTICS = True
    ENABLE_FEEDBACK = True
    ENABLE_CONVERSATION_HISTORY = True
    
    def __init__(self):
        """Initialize configuration and create necessary directories"""
        self._create_directories()
        self._validate_configuration()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.DOCS_FOLDER,
            self.PROMPTS_DIR,
            self.LOGS_DIR,
            os.path.dirname(self.MODEL_PATH),
            os.path.dirname(self.DATABASE_PATH)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _validate_configuration(self):
        """Validate configuration settings"""
        # Check if model file exists
        if not os.path.exists(self.MODEL_PATH):
            print(f"⚠️ Warning: Model file not found at {self.MODEL_PATH}")
            print("📥 Please download the model file to the models/ directory")
        
        # Check if docs folder has content
        if os.path.exists(self.DOCS_FOLDER):
            doc_files = list(Path(self.DOCS_FOLDER).glob("*.txt")) + list(Path(self.DOCS_FOLDER).glob("*.md"))
            if not doc_files:
                print(f"⚠️ Warning: No documents found in {self.DOCS_FOLDER}")
                print("📚 Consider adding financial documents to improve RAG performance")
    
    def get_model_config(self) -> dict:
        """Get model configuration as dictionary"""
        return {
            "model_path": self.MODEL_PATH,
            "n_ctx": self.MODEL_CONTEXT_SIZE,
            "n_threads": self.MODEL_N_THREADS,
            "n_batch": self.MODEL_N_BATCH,
            "temperature": self.MODEL_TEMPERATURE,
            "top_p": self.MODEL_TOP_P,
            "max_tokens": self.MODEL_MAX_TOKENS
        }
    
    def get_rag_config(self) -> dict:
        """Get RAG configuration as dictionary"""
        return {
            "docs_folder": self.DOCS_FOLDER,
            "embedding_model": self.EMBEDDING_MODEL,
            "chunk_size": self.RAG_CHUNK_SIZE,
            "chunk_overlap": self.RAG_CHUNK_OVERLAP,
            "top_k": self.RAG_TOP_K,
            "min_score": self.RAG_MIN_SCORE
        }
    
    def get_api_config(self) -> dict:
        """Get API configuration as dictionary"""
        return {
            "host": self.BACKEND_HOST,
            "port": self.BACKEND_PORT,
            "url": self.BACKEND_URL,
            "timeout": self.REQUEST_TIMEOUT,
            "cors_origins": self.CORS_ORIGINS
        }
    
    def get_ui_config(self) -> dict:
        """Get UI configuration as dictionary"""
        return {
            "theme": self.UI_THEME,
            "title": self.UI_TITLE,
            "description": self.UI_DESCRIPTION,
            "port": self.FRONTEND_PORT
        }
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    def get_log_level(self) -> str:
        """Get appropriate log level based on environment"""
        if self.is_production():
            return "WARNING"
        return self.LOG_LEVEL


# Create global config instance
config = Config()

# Print configuration info
if __name__ == "__main__":
    print("🔍 Financial Assistant AI Configuration")
    print(f"📁 Root directory: {ROOT_DIR}")
    print(f"📚 Docs folder: {config.DOCS_FOLDER}")
    print(f"📄 Model path: {config.MODEL_PATH}")
    print(f"🔧 Model exists: {os.path.exists(config.MODEL_PATH)}")
    print(f"📊 Docs exist: {os.path.exists(config.DOCS_FOLDER)}")
    if os.path.exists(config.DOCS_FOLDER):
        doc_files = list(Path(config.DOCS_FOLDER).glob("*.txt")) + list(Path(config.DOCS_FOLDER).glob("*.md"))
        print(f"📚 Document count: {len(doc_files)}")
    print(f"🌍 Environment: {'production' if config.is_production() else 'development'}")
    print(f"📝 Log level: {config.get_log_level()}")