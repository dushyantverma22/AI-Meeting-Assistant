import openai
import os
from typing import Dict, Any, Optional
from utils.logger import setup_logger
from utils.error_handler import (
    CorrectionError,
    APIError,
    ErrorSeverity,
    handle_error
)
from config.settings import config

logger = setup_logger("openai_context_corrector")

class OpenAIContextCorrector:
    """
    Context-aware text correction using OpenAI GPT models.
    Fixes domain-specific terminology and improves transcript quality.
    """

    SYSTEM_PROMPTS = {
        "financial": """You are an intelligent assistant specializing in financial terminology.
Your task is to process transcripts, ensuring financial terms are correctly formatted.
Examples:
- '401k' → '401(k) retirement savings plan'
- 'HSA' → 'Health Savings Account (HSA)'
- 'ROA' → 'Return on Assets (ROA)'
- 'LTV' → 'Loan-to-Value (LTV)'
Process the following transcript and maintain original meaning while fixing terminology:""",
        "medical": """You are an intelligent assistant specializing in medical terminology.
Ensure medical terms, abbreviations, and conditions are properly formatted and expanded.
Examples:
- 'HTN' → 'Hypertension (HTN)'
- 'DM' → 'Diabetes Mellitus (DM)'
- 'BP' → 'Blood Pressure (BP)'
Process the following medical transcript and expand abbreviations:""",
        "general": """You are an intelligent assistant specializing in transcription accuracy.
Correct common transcription errors while maintaining the original meaning.
Expand unclear abbreviations and fix terminology."""
    }

    def __init__(
        self,
        domain: str = "general",
        api_key: Optional[str] = None,
        model_name: str = "gpt-4",
        temperature: float = 0.2,
        max_tokens: int = 512
    ):
        """
        Initialize OpenAI-based context corrector.
        """
        self.domain = domain
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Set API key
        openai.api_key = self.api_key
        # Select prompt
        self.system_prompt = self.SYSTEM_PROMPTS.get(domain, self.SYSTEM_PROMPTS["general"])
        logger.info(f"OpenAIContextCorrector initialized for domain: {domain}")

    def correct(self, text: str) -> Dict[str, Any]:
        """
        Correct text using OpenAI GPT.
        """
        try:
            logger.info(f"Starting context correction for {len(text)} characters")
            prompt = f"{self.system_prompt}\n\n{text}\n\nCorrected Text:"
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=1,
                stop=None
            )
            corrected_text = response.choices[0].message['content'].strip()
            logger.info("Context correction completed successfully")
            return {
                "success": True,
                "original_text": text,
                "corrected_text": corrected_text,
                "model": self.model_name,
                "domain": self.domain
            }
        except Exception as e:
            error_info = handle_error(e, "Context correction", logger)
            raise CorrectionError(
                message="Context correction failed",
                error_code="OPENAI_CORRECTION_ERROR",
                severity=ErrorSeverity.HIGH,
                details=error_info
            )
