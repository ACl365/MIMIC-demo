"""
Logging utilities for the MIMIC project.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .config import get_project_root, load_config


def get_log_level_from_config() -> int:
    """
    Get the log level from the configuration file.
    
    Returns:
        int: Logging level (e.g., logging.INFO, logging.DEBUG)
    """
    try:
        config = load_config()
        log_level_str = config.get("logging", {}).get("level", "INFO")
        
        # Map string log levels to logging constants
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        return log_levels.get(log_level_str.upper(), logging.INFO)
    except Exception:
        # Default to INFO if there's an error loading the config
        return logging.INFO


def setup_logger(
    name: str = "mimic",
    log_level: Optional[int] = None,
    log_file: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and/or console handlers.
    
    Args:
        name (str, optional): Logger name. Defaults to "mimic".
        log_level (Optional[int], optional): Logging level. If None, gets from config.
            Defaults to None.
        log_file (Optional[str], optional): Path to log file. If None, logs to
            logs/{name}_{timestamp}.log. Defaults to None.
        console_output (bool, optional): Whether to output logs to console.
            Defaults to True.
    
    Returns:
        logging.Logger: Configured logger
    """
    # Get log level from config if not provided
    if log_level is None:
        log_level = get_log_level_from_config()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Add file handler if requested
    if log_file is not None or log_file != "":
        if log_file is None:
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(get_project_root(), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(logs_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = "mimic") -> logging.Logger:
    """
    Get a logger by name. If the logger doesn't exist, create it.
    
    Args:
        name (str, optional): Logger name. Defaults to "mimic".
    
    Returns:
        logging.Logger: Logger
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


def is_debug_enabled() -> bool:
    """
    Check if debug logging is enabled in the configuration.
    
    Returns:
        bool: True if debug logging is enabled, False otherwise
    """
    log_level = get_log_level_from_config()
    return log_level <= logging.DEBUG