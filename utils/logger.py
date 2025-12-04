"""
Logging configuration for AI Meeting Assistant
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_level: Logging level
        log_dir: Directory for log files
    
    Returns:
        Configured logger instance
    """
    try:
        # Create logs directory if it doesn't exist
        Path(log_dir).mkdir(exist_ok=True)
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # File handler
        log_file = Path(log_dir) / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    except Exception as e:
        print(f"Error setting up logger: {e}")
        # Fallback to basic logging
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        return logging.getLogger(name)
