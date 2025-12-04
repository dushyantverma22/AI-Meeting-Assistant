"""
AI Meeting Assistant - Gradio Web Application
Complete end-to-end meeting processing pipeline
"""

import gradio as gr
import traceback
from typing import Tuple, Optional
from pathlib import Path

from config.settings import config
from utils.logger import setup_logger
from utils.validators import validate_audio_file
from utils.error_handler import (
    handle_error,
    MeetingAssistantException,
    ErrorSeverity
)

# Import modules
from models.speech_to_text import WhisperTranscriber
from models.text_preprocessing import TextPreprocessor
from models.context_corrector import OpenAIContextCorrector
from models.meeting_summarizer import MeetingSummarizer
from chains.meeting_chain import MeetingProcessingChain

logger = setup_logger("gradio_app", config.app.log_level)


# Global instances (lazy loaded)
_transcriber = None
_preprocessor = None
_corrector = None
_summarizer = None
_chain = None


def initialize_models():
    """Initialize all models (lazy loading)"""
    global _transcriber, _preprocessor, _corrector, _summarizer, _chain
    
    try:
        if _transcriber is None:
            logger.info("Initializing models...")
            
            _transcriber = WhisperTranscriber(
                model_name=config.model.whisper_model
            )
            _preprocessor = TextPreprocessor()
            _corrector = OpenAIContextCorrector(domain="financial")
            _summarizer = MeetingSummarizer()
            _chain = MeetingProcessingChain(_summarizer.llm)
            
            logger.info("All models initialized successfully")
    
    except Exception as e:
        logger.critical(f"Model initialization failed: {e}")
        raise


def process_meeting_audio(
    audio_file: str,
    domain: str = "financial",
    include_correction: bool = True
) -> Tuple[str, str, str, str]:
    """
    Complete meeting processing pipeline
    
    Args:
        audio_file: Path to audio file
        domain: Domain for correction (financial, medical, general)
        include_correction: Whether to apply context correction
    
    Returns:
        Tuple of (transcript, cleaned_transcript, minutes, tasks)
    """
    try:
        # Initialize models
        initialize_models()
        
        # Step 1: Validate audio file
        validation = validate_audio_file(audio_file)
        if not validation["valid"]:
            return (
                f"Error: {validation['error']}",
                "",
                "",
                ""
            )
        
        # Step 2: Transcribe audio
        logger.info("Step 1: Transcribing audio...")
        gr.Info("🎙️ Transcribing audio... (This may take a moment)")
        
        transcription_result = _transcriber.transcribe(audio_file)
        if not transcription_result["success"]:
            return (
                f"Transcription failed: {transcription_result.get('error', 'Unknown error')}",
                "",
                "",
                ""
            )
        
        raw_transcript = transcription_result["text"]
        logger.info(f"Transcription complete: {len(raw_transcript)} characters")
        
        # Step 3: Preprocess text
        logger.info("Step 2: Preprocessing transcript...")
        gr.Info("📝 Cleaning up transcript...")
        
        preprocessing_result = _preprocessor.preprocess(raw_transcript)
        if not preprocessing_result["success"]:
            cleaned_transcript = raw_transcript
        else:
            cleaned_transcript = preprocessing_result["text"]
        
        logger.info("Preprocessing complete")
        
        # Step 4: Context correction (optional)
        corrected_transcript = cleaned_transcript
        if include_correction:
            try:
                logger.info("Step 3: Applying context correction...")
                gr.Info("🔧 Applying context-aware corrections...")
                
                correction_result = _corrector.correct(cleaned_transcript)
                if correction_result["success"]:
                    corrected_transcript = correction_result["corrected_text"]
                    logger.info("Context correction applied")
            
            except Exception as e:
                logger.warning(f"Context correction skipped: {e}")
                corrected_transcript = cleaned_transcript
        
        # Step 5: Generate meeting minutes and tasks
        logger.info("Step 4: Generating meeting minutes and tasks...")
        gr.Info("📊 Generating meeting minutes and task list...")
        
        try:
            # Generate using chain
            minutes_chain = _chain.create_minutes_chain()
            minutes = minutes_chain.invoke(corrected_transcript)
            
            # Generate tasks
            task_chain = _chain.create_task_chain()
            tasks = task_chain.invoke(corrected_transcript)
            
            logger.info("Meeting analysis complete")
            gr.Info("✅ Meeting analysis complete!")
        
        except Exception as e:
            logger.error(f"Chain execution failed: {e}")
            minutes = "Failed to generate meeting minutes"
            tasks = "Failed to generate task list"
        
        return (
            raw_transcript,
            corrected_transcript,
            minutes,
            tasks
        )
    
    except MeetingAssistantException as e:
        logger.error(f"Meeting processing error: {e.to_dict()}")
        return (
            f"Error: {e.message}",
            "",
            "",
            ""
        )
    
    except Exception as e:
        error_info = handle_error(e, "Meeting processing", logger)
        logger.error(f"Unexpected error: {error_info}")
        return (
            f"Error: {str(e)}",
            "",
            "",
            ""
        )


def save_results(
    transcript: str,
    minutes: str,
    tasks: str
) -> str:
    """
    Save results to file
    
    Args:
        transcript: Transcript text
        minutes: Meeting minutes
        tasks: Task list
    
    Returns:
        Path to saved file
    """
    try:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "meeting_analysis.txt"
        
        with open(output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("AI MEETING ASSISTANT - ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("TRANSCRIPT\n")
            f.write("-" * 80 + "\n")
            f.write(transcript + "\n\n")
            
            f.write("MEETING MINUTES\n")
            f.write("-" * 80 + "\n")
            f.write(minutes + "\n\n")
            
            f.write("TASK LIST\n")
            f.write("-" * 80 + "\n")
            f.write(tasks + "\n")
        
        logger.info(f"Results saved to {output_file}")
        return str(output_file)
    
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return "Failed to save results"


# Build Gradio Interface
with gr.Blocks(title="AI Meeting Assistant", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🎯 AI-Powered Meeting Assistant")
    gr.Markdown("Convert meeting audio to structured minutes and tasks automatically")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input")
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Upload Meeting Audio"
            )
            
            domain_selector = gr.Dropdown(
                choices=["financial", "medical", "general"],
                value="financial",
                label="Domain (for context correction)"
            )
            
            enable_correction = gr.Checkbox(
                value=True,
                label="Enable Context Correction"
            )
            
            process_btn = gr.Button(
                "🚀 Process Meeting",
                variant="primary",
                size="lg"
            )
        
        with gr.Column():
            gr.Markdown("### Output")
            
            with gr.Tabs():
                with gr.TabItem("Transcript"):
                    transcript_output = gr.Textbox(
                        label="Raw Transcript",
                        lines=10,
                        interactive=False
                    )
                
                with gr.TabItem("Cleaned Transcript"):
                    cleaned_output = gr.Textbox(
                        label="Processed Transcript",
                        lines=10,
                        interactive=False
                    )
                
                with gr.TabItem("Meeting Minutes"):
                    minutes_output = gr.Textbox(
                        label="Meeting Minutes",
                        lines=10,
                        interactive=False
                    )
                
                with gr.TabItem("Task List"):
                    tasks_output = gr.Textbox(
                        label="Action Items",
                        lines=10,
                        interactive=False
                    )
    
    with gr.Row():
        download_btn = gr.Button("📥 Download All Results")
    
    # Connect buttons
    process_btn.click(
        process_meeting_audio,
        inputs=[audio_input, domain_selector, enable_correction],
        outputs=[transcript_output, cleaned_output, minutes_output, tasks_output]
    )
    
    download_btn.click(
        save_results,
        inputs=[transcript_output, minutes_output, tasks_output],
        outputs=gr.Textbox(label="Status", interactive=False)
    )
    
    gr.Markdown("""
    ---
    ### How It Works
    1. **Upload Audio** - Provide meeting recording (MP3, WAV, etc.)
    2. **Select Domain** - Choose domain for context-aware corrections
    3. **Process** - AI will:
       - Transcribe audio using Whisper
       - Clean and preprocess text
       - Apply context-aware corrections
       - Generate meeting minutes
       - Create actionable task list
    4. **Download** - Save complete analysis as text file
    
    ### Best Practices
    - Use clear, high-quality audio
    - Keep meetings under 30 minutes for best results
    - Select appropriate domain for accurate corrections
    """)


if __name__ == "__main__":
    logger.info("Starting AI Meeting Assistant")
    logger.info(f"Configuration: {config}")
    
    interface.launch(
        server_name=config.app.server_name,
        server_port=config.app.server_port,
        share=config.app.share
    )
