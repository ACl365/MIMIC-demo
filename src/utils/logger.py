"""
Logging utilities for the MIMIC project.

Provides functions to configure and retrieve logger instances based on settings
defined in the project's configuration file (`config.yaml`). Handles log levels,
file output, and console output.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional  # Added Dict, Any for config type hint

# Import config functions inside methods to avoid circular imports during startup
# from .config import get_project_root, load_config


def get_log_level_from_config() -> int:
    """
    Get the logging level (e.g., logging.INFO, logging.DEBUG) from the configuration file.

    Reads the 'logging.level' setting from the config. Defaults to logging.INFO
    if the setting is missing, invalid, or if the config file cannot be loaded.

    Returns:
        int: The logging level constant (e.g., logging.INFO, logging.DEBUG).
    """
    try:
        from .config import load_config  # Import here to avoid circular dependency

        config = load_config()
        # Use .get() with defaults for safer access
        log_level_str = config.get("logging", {}).get("level", "INFO").upper()

        # Map string log levels to logging constants
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        level = log_levels.get(log_level_str, logging.INFO)
        # Ensure the retrieved level is actually an int (logging level constant)
        if not isinstance(level, int):
            print(
                f"WARNING: Invalid log level string '{log_level_str}' resolved to non-integer. Defaulting to INFO."
            )
            return logging.INFO
        return level

    except Exception as e:
        # Default to INFO if there's any error loading/parsing the config
        print(
            f"WARNING: Could not load log level from config ({e}). Defaulting to INFO."
        )
        return logging.INFO


def setup_logger(
    name: str = "mimic",
    log_level: Optional[int] = None,
    log_file: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up and configure a logger instance.

    Configures a logger with the specified name, level, and handlers.
    If `log_level` is None, it attempts to get the level from the project config.
    If `log_file` is None and file output is desired (implicitly or via config),
    it creates a timestamped log file in the 'logs/' directory.

    Args:
        name (str, optional): Name of the logger. Defaults to "mimic".
        log_level (Optional[int], optional): Logging level (e.g., logging.DEBUG).
                                             If None, determined by `get_log_level_from_config()`.
                                             Defaults to None.
        log_file (Optional[str], optional): Explicit path to the log file. If set to an empty
                                            string "", file logging is disabled. If set to None,
                                            a default timestamped log file is created in 'logs/'.
                                            Defaults to None.
        console_output (bool, optional): Whether to add a handler for console output (stdout).
                                         Defaults to True.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Get log level from config if not provided
    if log_level is None:
        log_level = get_log_level_from_config()

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers if logger already exists and is configured
    if logger.handlers:
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # Add date format
    )

    # Determine if file output is enabled based on log_file argument
    # None means use default path, "" means disable, specific path means use that path.
    enable_file_output = log_file is None or log_file != ""

    # Add file handler if enabled
    if enable_file_output:
        actual_log_file_path = log_file  # Use provided path if not None
        if actual_log_file_path is None:
            # Create default log file path if log_file was None
            try:
                from .config import get_project_root  # Import here

                logs_dir = get_project_root() / "logs"
                logs_dir.mkdir(exist_ok=True)  # Use pathlib's mkdir

                # Create log file with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                actual_log_file_path = logs_dir / f"{name}_{timestamp}.log"
            except Exception as e:
                print(f"ERROR: Could not create default log file path: {e}")
                actual_log_file_path = (
                    None  # Disable file logging if path creation fails
                )

        if actual_log_file_path:
            try:
                file_handler = logging.FileHandler(
                    str(actual_log_file_path), encoding="utf-8"
                )  # Specify encoding
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                # print(f"Logging to file: {actual_log_file_path}") # Optional: confirm log file path
            except Exception as e:
                print(
                    f"ERROR: Could not set up file handler for {actual_log_file_path}: {e}"
                )

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Prevent propagation to root logger if handlers are added
    logger.propagate = False

    return logger


def get_logger(name: str = "mimic") -> logging.Logger:
    """
    Get a logger instance by name.

    If the logger with the given name already exists and has handlers, it's returned directly.
    Otherwise, it sets up a new logger based on the project configuration (log level,
    file output, console output). This ensures consistent logger configuration across the project.

    Args:
        name (str, optional): Name of the logger. Defaults to "mimic".

    Returns:
        logging.Logger: The logger instance, potentially newly configured.
    """
    logger_instance = logging.getLogger(name)

    # If logger has no handlers, set it up based on config
    if not logger_instance.handlers:
        # Determine setup based on config (if available) or defaults
        try:
            from .config import load_config

            config = load_config()
            log_config = config.get("logging", {})
            file_output_enabled = log_config.get("file_output", True)
            console_output_enabled = log_config.get("console_output", True)
            # Use default log file path logic within setup_logger if file_output is True
            # Pass None to setup_logger to trigger default path generation if file output is enabled
            # Pass "" to setup_logger explicitly disable file logging
            log_file_path = None if file_output_enabled else ""
            # Retrieve level from config via helper function
            level = get_log_level_from_config()

            # Call setup_logger with parameters derived from config
            logger_instance = setup_logger(
                name,
                log_level=level,
                console_output=console_output_enabled,
                log_file=log_file_path,
            )
        except Exception as e:
            # Fallback to basic setup if config loading or setup fails
            print(
                f"WARNING: Error setting up logger from config ({e}). Using default setup."
            )
            logger_instance = setup_logger(name)  # Basic setup with defaults

    return logger_instance


def is_debug_enabled() -> bool:
    """
    Check if DEBUG logging level is enabled based on the configuration.

    Useful for conditionally executing expensive debug logging or operations.

    Returns:
        bool: True if the configured log level is DEBUG or lower, False otherwise.
    """
    # Note: get_log_level_from_config already imports load_config internally
    log_level = get_log_level_from_config()
    return log_level <= logging.DEBUG
