"""
Text preprocessing module
Cleans and normalizes transcribed text
"""

import re
import string
from typing import Dict, Any, List

from utils.logger import setup_logger
from utils.error_handler import PreprocessingError, ErrorSeverity, handle_error


logger = setup_logger("preprocessing")


class TextPreprocessor:
    """
    Text preprocessing and cleanup
    
    Handles:
    - Removing non-ASCII characters
    - Fixing common transcription errors
    - Normalizing whitespace
    - Adding proper punctuation
    """
    
    def __init__(self):
        """Initialize preprocessor"""
        self.non_ascii_pattern = re.compile(r'[^\x00-\x7F]+')
        logger.info("TextPreprocessor initialized")
    
    def remove_non_ascii(self, text: str) -> str:
        """
        Remove non-ASCII characters
        
        Args:
            text: Input text
        
        Returns:
            Text with only ASCII characters
        """
        try:
            return ''.join(char for char in text if ord(char) < 128)
        except Exception as e:
            handle_error(e, "Remove non-ASCII", logger)
            raise PreprocessingError(
                message="Failed to remove non-ASCII characters",
                error_code="ASCII_REMOVAL_ERROR",
                severity=ErrorSeverity.MEDIUM
            )
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize multiple spaces and line breaks
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        try:
            # Replace multiple spaces with single space
            text = re.sub(r' +', ' ', text)
            # Replace multiple newlines with double newline
            text = re.sub(r'\n\n+', '\n\n', text)
            return text.strip()
        except Exception as e:
            handle_error(e, "Normalize whitespace", logger)
            raise PreprocessingError(
                message="Failed to normalize whitespace",
                error_code="WHITESPACE_NORMALIZATION_ERROR",
                severity=ErrorSeverity.MEDIUM
            )
    
    def fix_common_errors(self, text: str) -> str:
        """
        Fix common transcription errors
        
        Args:
            text: Input text
        
        Returns:
            Text with common errors fixed
        """
        try:
            # Common transcription error replacements
            replacements = {
                r'\bthe the\b': 'the',
                r'\band and\b': 'and',
                r'\bto to\b': 'to',
                r'\ba a\b': 'a',
                r'\ban an\b': 'an',
                r'\bis is\b': 'is',
                r'\bwas was\b': 'was',
            }
            
            for pattern, replacement in replacements.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
            return text
        except Exception as e:
            handle_error(e, "Fix common errors", logger)
            raise PreprocessingError(
                message="Failed to fix common transcription errors",
                error_code="ERROR_FIXING_ERROR",
                severity=ErrorSeverity.MEDIUM
            )
    
    def preprocess(
        self,
        text: str,
        remove_ascii: bool = True,
        normalize_spaces: bool = True,
        fix_errors: bool = True
    ) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text to preprocess
            remove_ascii: Remove non-ASCII characters
            normalize_spaces: Normalize whitespace
            fix_errors: Fix common errors
        
        Returns:
            Dictionary with preprocessing results
        """
        try:
            logger.info("Starting text preprocessing")
            
            processed_text = text
            
            if remove_ascii:
                processed_text = self.remove_non_ascii(processed_text)
            
            if normalize_spaces:
                processed_text = self.normalize_whitespace(processed_text)
            
            if fix_errors:
                processed_text = self.fix_common_errors(processed_text)
            
            logger.info("Text preprocessing completed successfully")
            
            return {
                "success": True,
                "original_length": len(text),
                "processed_length": len(processed_text),
                "text": processed_text
            }
        
        except PreprocessingError:
            raise
        
        except Exception as e:
            error_info = handle_error(e, "Text preprocessing", logger)
            raise PreprocessingError(
                message="Text preprocessing pipeline failed",
                error_code="PREPROCESSING_PIPELINE_ERROR",
                severity=ErrorSeverity.HIGH,
                details=error_info
            )
