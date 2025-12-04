"""
LangChain integration for meeting processing pipeline (OpenAI version)
Orchestrates the flow from transcript to meeting minutes, tasks, and analysis
"""

from typing import Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from utils.logger import setup_logger
from utils.error_handler import (
    handle_error,
    MeetingAssistantException
)
from config.settings import config

logger = setup_logger("meeting_chain")

class MeetingProcessingChain:
    """
    LangChain-based meeting processing pipeline (OpenAI)
    
    Orchestrates:
    - Prompt template creation
    - Chain composition using pipe operator
    - Output parsing
    """
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7, max_tokens: int = 1500):
        """
        Initialize chain with OpenAI model parameters.
        """
        try:
            # Initialize OpenAI LLM
            self.llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=None  # Will be set dynamically or via environment
            )
            self.output_parser = StrOutputParser()
            logger.info(f"MeetingProcessingChain initialized with model: {model_name}")
        except Exception as e:
            handle_error(e, "Chain initialization", logger)
            raise

    def create_minutes_chain(self) -> Any:
        """
        Create chain for meeting minutes generation.
        """
        try:
            template = """You are an expert meeting analyst. 
            
Generate comprehensive meeting minutes from the following transcript.

Include:
- Executive Summary (2-3 sentences)
- Key Points Discussed (bullet points)
- Decisions Made (clear outcomes)
- Important Announcements
- Follow-up Items

Transcript:
{context}

Meeting Minutes:"""
            prompt = ChatPromptTemplate.from_template(template)
            chain = (
                {"context": RunnablePassthrough()}
                | prompt
                | self.llm
                | self.output_parser
            )
            logger.info("Minutes chain created successfully")
            return chain
        except Exception as e:
            handle_error(e, "Create minutes chain", logger)
            raise

    def create_task_chain(self) -> Any:
        """
        Create chain for task list generation.
        """
        try:
            template = """You are an expert in project management and task identification.

Extract and organize all actionable items from the following transcript.

For each task, provide:
1. Task Description (clear and specific)
2. Assignee/Owner (who is responsible)
3. Deadline (when should it be completed)
4. Priority (High/Medium/Low)
5. Dependencies (any blocking items)

Format as a numbered list with clear separations.

Transcript:
{context}

Task List:"""
            prompt = ChatPromptTemplate.from_template(template)
            chain = (
                {"context": RunnablePassthrough()}
                | prompt
                | self.llm
                | self.output_parser
            )
            logger.info("Task chain created successfully")
            return chain
        except Exception as e:
            handle_error(e, "Create task chain", logger)
            raise

    def create_comprehensive_chain(self) -> Any:
        """
        Create comprehensive chain for full meeting analysis.
        """
        try:
            template = """You are an expert meeting facilitator and business analyst.

Analyze the following meeting transcript comprehensively.

Provide:
1. MEETING SUMMARY (2-3 sentences about the overall meeting)

2. KEY POINTS DISCUSSED
   - List 3-5 main topics covered

3. DECISIONS MADE
   - Important outcomes and conclusions

4. ACTION ITEMS
   - Task, Owner, Deadline, Priority for each item

5. KEY TAKEAWAYS
   - Top 2-3 takeaway messages

Transcript:
{context}

Comprehensive Meeting Analysis:"""
            prompt = ChatPromptTemplate.from_template(template)
            chain = (
                {"context": RunnablePassthrough()}
                | prompt
                | self.llm
                | self.output_parser
            )
            logger.info("Comprehensive chain created successfully")
            return chain
        except Exception as e:
            handle_error(e, "Create comprehensive chain", logger)
            raise

    def execute_chain(self, chain: Any, input_text: str) -> Dict[str, Any]:
        """
        Execute a chain with input text, with exception handling.
        """
        try:
            logger.info(f"Executing chain with input length: {len(input_text)}")
            result = chain.invoke({"context": input_text})
            logger.info("Chain execution successful")
            return {
                "success": True,
                "output": result,
                "input_length": len(input_text)
            }
        except Exception as e:
            error_info = handle_error(e, "Chain execution", logger)
            return {
                "success": False,
                "error": str(e),
                "details": error_info
            }
