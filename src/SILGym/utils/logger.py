"""
Unified logging system for SILGym.

This module provides a centralized logging system that:
- Captures all logs from program start
- Shows function/module context for each log
- Saves logs to experiment path once available
- Maintains consistency across all modules
"""

import logging
import sys
import os
import inspect
from datetime import datetime
from typing import Optional, List, Tuple
from pathlib import Path
import threading
from collections import deque


class BufferedFileHandler(logging.Handler):
    """
    A file handler that buffers logs until a file path is available.
    
    This handler stores logs in memory until set_log_path() is called,
    then writes all buffered logs to the file and continues logging there.
    """
    
    def __init__(self):
        super().__init__()
        self.buffer: deque = deque(maxlen=10000)  # Buffer up to 10k messages
        self.file_handler: Optional[logging.FileHandler] = None
        self.log_path: Optional[str] = None
        self._lock = threading.Lock()
        
    def emit(self, record):
        """Emit a log record - buffer it or write to file if available."""
        with self._lock:
            if self.file_handler is None:
                # Buffer the formatted record
                self.buffer.append(self.format(record))
            else:
                # Write to file
                self.file_handler.emit(record)
                
    def set_log_path(self, log_path: str):
        """Set the log file path and flush buffered logs."""
        with self._lock:
            if self.file_handler is not None:
                return  # Already set
                
            self.log_path = log_path
            
            # Create directory if needed
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            # Create file handler
            self.file_handler = logging.FileHandler(log_path, mode='w')
            self.file_handler.setFormatter(self.formatter)
            
            # Write header
            with open(log_path, 'w') as f:
                f.write(f"{'='*80}\n")
                f.write(f"SILGym Experiment Log\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
            
            # Flush buffered logs
            with open(log_path, 'a') as f:
                for log_line in self.buffer:
                    f.write(log_line + '\n')
            
            self.buffer.clear()
            
    def close(self):
        """Close the file handler if it exists."""
        with self._lock:
            if self.file_handler:
                self.file_handler.close()
                

class ContextualFormatter(logging.Formatter):
    """
    Custom formatter that includes function and module context.
    """
    
    def format(self, record):
        # Get the caller's frame info
        if not hasattr(record, 'funcName') or record.funcName == '<module>':
            # Try to get better context
            frame = inspect.currentframe()
            for _ in range(10):  # Look up to 10 frames back
                if frame is None:
                    break
                frame = frame.f_back
                if frame and frame.f_code.co_name not in ['emit', 'format', 'info', 'debug', 'warning', 'error', 'critical']:
                    record.funcName = frame.f_code.co_name
                    record.pathname = frame.f_code.co_filename
                    record.lineno = frame.f_lineno
                    break
        
        # Extract module name from pathname
        if hasattr(record, 'pathname'):
            module_parts = []
            path_parts = Path(record.pathname).parts
            
            # Find 'SILGym' or 'exp' in path and include from there
            start_idx = -1
            for i, part in enumerate(path_parts):
                if part in ['SILGym', 'exp', 'src']:
                    start_idx = i
                    break
            
            if start_idx >= 0:
                module_parts = path_parts[start_idx:]
                # Remove .py extension from last part
                if module_parts and module_parts[-1].endswith('.py'):
                    module_parts = list(module_parts)
                    module_parts[-1] = module_parts[-1][:-3]
            else:
                # Fallback to just filename
                module_parts = [Path(record.pathname).stem]
                
            record.module_context = '.'.join(module_parts)
        else:
            record.module_context = record.module
            
        # Format the message
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level = record.levelname.ljust(8)
        module_func = f"{record.module_context}:{record.funcName}:{record.lineno}"
        
        # Truncate module_func if too long
        if len(module_func) > 50:
            module_func = '...' + module_func[-47:]
        else:
            module_func = module_func.ljust(50)
            
        message = record.getMessage()
        
        return f"[{timestamp}] [{level}] [{module_func}] {message}"


class SILGymLogger:
    """
    Singleton logger manager for SILGym.
    
    This class manages the global logger configuration and provides
    convenience methods for logging throughout the application.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._setup_logger()
            
    def _setup_logger(self):
        """Set up the root logger with console and buffered file handlers."""
        # Get root logger
        self.logger = logging.getLogger('SILGym')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        formatter = ContextualFormatter()
        
        # Console handler - less verbose
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Buffered file handler - full details
        self.file_handler = BufferedFileHandler()
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(self.file_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
        
    def set_experiment_path(self, exp_path: str):
        """
        Set the experiment path and create log file.
        
        Args:
            exp_path: Experiment directory path
        """
        log_file = os.path.join(exp_path, 'experiment.log')
        self.file_handler.set_log_path(log_file)
        self.logger.info(f"Experiment path set to: {exp_path}")
        
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name (typically __name__ from the calling module)
            
        Returns:
            Logger instance
        """
        if name:
            return logging.getLogger(f'SILGym.{name}')
        return self.logger
        
    @staticmethod
    def shutdown():
        """Shutdown the logging system cleanly."""
        instance = SILGymLogger()
        instance.file_handler.close()
        logging.shutdown()


# Convenience functions for direct logging
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return SILGymLogger().get_logger(name)


def set_experiment_path(exp_path: str):
    """
    Set the experiment path for logging.
    
    Args:
        exp_path: Experiment directory path
    """
    SILGymLogger().set_experiment_path(exp_path)


def log_info(message: str, logger_name: Optional[str] = None):
    """Log an info message."""
    logger = get_logger(logger_name)
    logger.info(message)


def log_debug(message: str, logger_name: Optional[str] = None):
    """Log a debug message."""
    logger = get_logger(logger_name)
    logger.debug(message)


def log_warning(message: str, logger_name: Optional[str] = None):
    """Log a warning message."""
    logger = get_logger(logger_name)
    logger.warning(message)


def log_error(message: str, logger_name: Optional[str] = None):
    """Log an error message."""
    logger = get_logger(logger_name)
    logger.error(message)


def log_critical(message: str, logger_name: Optional[str] = None):
    """Log a critical message."""
    logger = get_logger(logger_name)
    logger.critical(message)


# Initialize logger on import
_logger_instance = SILGymLogger()