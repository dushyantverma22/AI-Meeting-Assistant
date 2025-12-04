"""
Meeting summarization module using OpenAI GPT models
Generates meeting minutes and task lists
"""

from typing import Dict, Any, Optional
import os
from langchain_openai import ChatOpenAI
from utils.logger import setup_logger
from utils.error_handler import (
    SummarizationError,
    APIError,
    ErrorSeverity,
    handle_error
)
from config.settings import config

# Initialize logger
logger = setup_logger("meeting_summarizer")

class MeetingSummarizer:
    """
    Meeting summarization using OpenAI GPT models
    
    Generates:
    - Meeting minutes (key points, decisions)
    - Task lists (action items, assignees, deadlines)
    - Key takeaways
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",  # default model, can be overridden
        temperature: float = 0.7,
        max_tokens: int = 500,
        api_key: Optional[str] = None
    ):
        """
        Initialize the OpenAI-based meeting summarizer.
        Args:
            model_name: OpenAI model name (e.g., gpt-3.5-turbo, gpt-4)
            temperature: Sampling temperature
            max_tokens: Max tokens for output
            api_key: OpenAI API key (fallback to env if None)
        """
        try:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not provided or set in environment.")
            self.model_name = model_name
            self.temperature = temperature
            self.max_tokens = max_tokens

            # Initialize the ChatOpenAI model
            self.llm = self._initialize_llm()
            logger.info(f"OpenAI MeetingSummarizer initialized with model: {self.model_name}")
        except Exception as e:
            error_info = handle_error(e, "Initialization", logger)
            raise APIError(
                message="Failed to initialize OpenAI MeetingSummarizer",
                error_code="OPENAI_INIT_ERROR",
                severity=ErrorSeverity.CRITICAL,
                details=error_info
            )

    def _initialize_llm(self) -> ChatOpenAI:
        """Create and return the ChatOpenAI model instance with error handling."""
        try:
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=self.api_key
            )
        except Exception as e:
            error_info = handle_error(e, "LLM Initialization", logger)
            raise APIError(
                message="Failed to initialize OpenAI LLM",
                error_code="OPENAI_LLM_ERROR",
                severity=ErrorSeverity.CRITICAL,
                details=error_info
            )

    def generate_meeting_minutes(self, transcript: str) -> Dict[str, Any]:
        """
        Generate structured meeting minutes from transcript.
        Args:
            transcript: Meeting transcript text
        Returns:
            Dict with success status and meeting minutes
        """
        try:
            logger.info("Generating meeting minutes")
            prompt = (
                f"Based on the following meeting transcript, generate structured meeting minutes including:\n"
                "1. Key Points Discussed - Main topics covered\n"
                "2. Decisions Made - Important decisions and outcomes\n"
                "3. Attendees (if mentioned)\n"
                "4. Date/Time (if mentioned)\n\n"
                f"Transcript:\n{transcript}\n\n"
                "Generate clear, concise meeting minutes:"
            )
            minutes = self.llm.invoke(prompt)
            logger.info("Meeting minutes generated successfully")
            return {
                "success": True,
                "meeting_minutes": minutes,
                "type": "meeting_minutes"
            }
        except Exception as e:
            error_info = handle_error(e, "Generate meeting minutes", logger)
            raise SummarizationError(
                message="Failed to generate meeting minutes",
                error_code="MINUTES_GENERATION_ERROR",
                severity=ErrorSeverity.HIGH,
                details=error_info
            )

    def generate_task_list(self, transcript: str) -> Dict[str, Any]:
        """
        Generate action task list from transcript.
        Args:
            transcript: Meeting transcript text
        Returns:
            Dict with success status and task list
        """
        try:
            logger.info("Generating task list")
            prompt = (
                f"Based on the following meeting transcript, generate an actionable task list.\n"
                "For each task, specify:\n"
                "1. Task Description\n"
                "2. Assignee (if mentioned, otherwise 'TBD')\n"
                "3. Deadline (if mentioned, otherwise 'Not specified')\n"
                "4. Priority (High/Medium/Low)\n\n"
                f"Transcript:\n{transcript}\n\n"
                "Generate the task list:"
            )
            tasks = self.llm.invoke(prompt)
            logger.info("Task list generated successfully")
            return {
                "success": True,
                "task_list": tasks,
                "type": "task_list"
            }
        except Exception as e:
            error_info = handle_error(e, "Generate task list", logger)
            raise SummarizationError(
                message="Failed to generate task list",
                error_code="TASK_LIST_ERROR",
                severity=ErrorSeverity.HIGH,
                details=error_info
            )

    def summarize_meeting(self, transcript: str) -> Dict[str, Any]:
        """
        Generate full meeting summary (minutes + tasks).
        Args:
            transcript: Meeting transcript text
        Returns:
            Dict with all results
        """
        try:
            logger.info("Starting complete meeting summarization")
            minutes_result = self.generate_meeting_minutes(transcript)
            tasks_result = self.generate_task_list(transcript)
            return {
                "success": True,
                "meeting_minutes": minutes_result["meeting_minutes"],
                "task_list": tasks_result["task_list"],
                "transcript_length": len(transcript),
                "model": self.model_name
            }
        except SummarizationError:
            raise
        except Exception as e:
            error_info = handle_error(e, "Complete meeting summarization", logger)
            raise SummarizationError(
                message="Meeting summarization pipeline failed",
                error_code="SUMMARIZATION_PIPELINE_ERROR",
                severity=ErrorSeverity.HIGH,
                details=error_info
            )
