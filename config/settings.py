import os
from dotenv import load_dotenv
from dataclasses import dataclass

@dataclass
class APIConfig:
    OPENAI_API_KEY: str

class ModelConfig:

     # Speech-to-Text
    whisper_model: str = "medium"

    # Text Correction (Llama 3.2)
    correction_model: str = "gpt-4o-mini"
    correction_temperature: float = 0.2
    correction_top_p: float = 0.6
    
    # Summarization (IBM Granite)
    summarizer_model: str = "gpt-4o-mini"
    summarizer_temperature: float = 0.5
    summarizer_max_tokens: int = 512
    summarizer_min_tokens: int = 1
    summarizer_top_k: int = 50
    summarizer_top_p: float = 1.0

@dataclass
class AppConfig:
    """Application configuration"""
    server_name: str = "localhost"
    server_port: int = 7860
    share: bool = False
    debug: bool = True
    log_level: str = "INFO"


class config:
    """Main configuration class"""
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # API Configuration
        self.api = APIConfig(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"))

        # Model Configuration
        self.model = ModelConfig()

        # Application Configuration
        self.app = AppConfig()

    def validate(self):
        """Validate configuration settings"""
        if not self.api.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")

# Instantiate and validate configuration
config = config()  


