"""
Input validation utilities
"""

import os
from pathlib import Path
from typing import Dict, Any


SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 500MB


def validate_audio_file(file_path: str) -> Dict[str, Any]:
    """
    Validate audio file
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Validation result dictionary
    """
    try:
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            return {
                "valid": False,
                "error": f"File does not exist: {file_path}"
            }
        
        # Check file extension
        if path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
            return {
                "valid": False,
                "error": f"Unsupported file format. Supported: {SUPPORTED_AUDIO_FORMATS}"
            }
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return {
                "valid": False,
                "error": f"File too large. Max size: {MAX_FILE_SIZE} bytes"
            }
        
        return {
            "valid": True,
            "file_path": str(path.absolute()),
            "file_size": file_size,
            "file_format": path.suffix.lower()
        }
    
    except Exception as e:
        return {
            "valid": False,
            "error": f"Validation error: {str(e)}"
        }


def validate_text_input(text: str, min_length: int = 10) -> Dict[str, Any]:
    """
    Validate text input
    
    Args:
        text: Text to validate
        min_length: Minimum text length
    
    Returns:
        Validation result dictionary
    """
    if not text:
        return {"valid": False, "error": "Text is empty"}
    
    if len(text) < min_length:
        return {
            "valid": False,
            "error": f"Text too short. Minimum length: {min_length}"
        }
    
    return {"valid": True, "text_length": len(text)}
