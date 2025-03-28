# core/utils/logging.py
import logging
import os
import sys
from typing import Optional

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a logger with the specified name and level.
    
    Args:
        name: Logger name
        level: Logging level (defaults to INFO)
        
    Returns:
        Logger instance
    """
    if level is None:
        level = logging.INFO
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Add console handler if it doesn't already exist
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler for persistent logging if LOG_DIR is set
        log_dir = os.environ.get("LOG_DIR")
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, f"{name.replace('.', '_')}.log")
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger