"""
Speech-to-Text module using OpenAI Whisper
Handles audio file transcription with error handling
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
import whisper

from utils.logger import setup_logger
from utils.error_handler import (
    TranscriptionError,
    AudioProcessingError,
    ErrorSeverity,
    handle_error
)
from utils.validators import validate_audio_file


logger = setup_logger("speech_to_text")


class WhisperTranscriber:
    """
    Speech-to-Text transcriber using OpenAI Whisper
    
    Attributes:
        model_name: Whisper model to use (base, small, medium, large)
        device: Device to run model on (cuda, cpu)
        model: Loaded Whisper model
    """
    
    def __init__(self, model_name: str = "small", device: Optional[str] = None):
        """
        Initialize Whisper transcriber
        
        Args:
            model_name: Model size (base, small, medium, large)
            device: Device to use (auto-detect if None)
        
        Raises:
            TranscriptionError: If model loading fails
        """
        try:
            self.model_name = model_name
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            
            logger.info(f"Loading Whisper model: {model_name} on {self.device}")
            self.model = whisper.load_model(model_name, device=self.device)
            logger.info(f"Successfully loaded Whisper model: {model_name}")
        
        except Exception as e:
            error_info = handle_error(
                e,
                "Whisper model initialization",
                logger
            )
            raise TranscriptionError(
                message=f"Failed to load Whisper model: {str(e)}",
                error_code="WHISPER_LOAD_ERROR",
                severity=ErrorSeverity.CRITICAL,
                details=error_info
            )
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es'). Auto-detect if None
        
        Returns:
            Dictionary with transcription results
        
        Raises:
            AudioProcessingError: If audio file validation fails
            TranscriptionError: If transcription fails
        """
        try:
            # Validate audio file
            validation_result = validate_audio_file(audio_path)
            if not validation_result["valid"]:
                raise AudioProcessingError(
                    message=validation_result["error"],
                    error_code="INVALID_AUDIO_FILE",
                    severity=ErrorSeverity.HIGH,
                    details={"file": audio_path}
                )
            
            logger.info(f"Starting transcription for: {audio_path}")
            
            # Load and process audio
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Detect language if not specified
            if language is None:
                _, probs = self.model.detect_language(mel)
                language = max(probs, key=probs.get)
                logger.info(f"Detected language: {language}")
            
            # Decode audio
            options = whisper.DecodingOptions(
                language=language,
                fp16=torch.cuda.is_available()
            )
            result = whisper.decode(self.model, mel, options)
            
            logger.info("Transcription completed successfully")
            
            return {
                "success": True,
                "text": result.text,
                "language": language,
                "segments": result.segments if hasattr(result, 'segments') else [],
                "no_speech_prob": result.no_speech_prob if hasattr(result, 'no_speech_prob') else 0
            }
        
        except AudioProcessingError:
            raise
        
        except Exception as e:
            error_info = handle_error(e, "Audio transcription", logger)
            raise TranscriptionError(
                message=f"Transcription failed: {str(e)}",
                error_code="TRANSCRIPTION_ERROR",
                severity=ErrorSeverity.HIGH,
                details=error_info
            )
    
    def transcribe_batch(
        self,
        audio_paths: list
    ) -> Dict[str, Any]:
        """
        Transcribe multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
        
        Returns:
            Dictionary with batch transcription results
        """
        results = {
            "success": True,
            "transcriptions": [],
            "errors": []
        }
        
        for audio_path in audio_paths:
            try:
                result = self.transcribe(audio_path)
                results["transcriptions"].append({
                    "file": audio_path,
                    **result
                })
            except Exception as e:
                results["success"] = False
                results["errors"].append({
                    "file": audio_path,
                    "error": str(e)
                })
                logger.error(f"Failed to transcribe {audio_path}: {e}")
        
        return results
