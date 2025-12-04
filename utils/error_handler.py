"""
Custom exception classes and error handling for AI Meeting Assistant
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MeetingAssistantException(Exception):
    """Base exception for Meeting Assistant"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging"""
        return {
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "details": self.details
        }


class AudioProcessingError(MeetingAssistantException):
    """Raised when audio processing fails"""
    pass


class TranscriptionError(MeetingAssistantException):
    """Raised when speech-to-text conversion fails"""
    pass


class PreprocessingError(MeetingAssistantException):
    """Raised when text preprocessing fails"""
    pass


class CorrectionError(MeetingAssistantException):
    """Raised when context correction fails"""
    pass


class SummarizationError(MeetingAssistantException):
    """Raised when summarization fails"""
    pass


class APIError(MeetingAssistantException):
    """Raised when API calls fail"""
    pass


def handle_error(
    exception: Exception,
    context: str,
    logger=None
) -> Dict[str, Any]:
    """
    Centralized error handling
    
    Args:
        exception: The exception that occurred
        context: Context where error occurred
        logger: Logger instance
    
    Returns:
        Error information dictionary
    """
    error_info = {
        "success": False,
        "error": str(exception),
        "context": context,
        "type": type(exception).__name__
    }
    
    if isinstance(exception, MeetingAssistantException):
        error_info.update(exception.to_dict())
    
    if logger:
        if isinstance(exception, MeetingAssistantException):
            if exception.severity == ErrorSeverity.CRITICAL:
                logger.critical(f"{context}: {error_info}")
            else:
                logger.error(f"{context}: {error_info}")
        else:
            logger.error(f"{context}: {error_info}")
    
    return error_info
