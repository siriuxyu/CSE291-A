"""Utility & helper functions."""

import json
import logging
import functools
import time
from datetime import datetime
from pathlib import Path
from langchain_core.messages import BaseMessage


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'function'):
            log_data['function'] = record.function
        if hasattr(record, 'details'):
            log_data['details'] = record.details
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_file_name: str = None
) -> logging.Logger:
    """Set up logging configuration for the application.
    
    This function configures logging to output to both console and file.
    Log files are automatically created in the specified directory.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        log_file_name: Optional custom log file name. If None, uses timestamp.
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Generate log file name with timestamp if not provided
    if log_file_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"react_agent_{timestamp}.log"
    
    log_file_path = log_path / log_file_name
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Structured JSON formatter for file
    structured_formatter = StructuredFormatter()
    
    # File handler - logs everything with structured JSON format
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(structured_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - logs INFO and above with simple format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info("Logging initialized", extra={
        'log_file': str(log_file_path),
        'log_level': log_level
    })
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_call(logger: logging.Logger, request_id: str = None, user_id: str = None):
    """Decorator to log function calls with timing information.
    
    Args:
        logger: Logger instance to use
        request_id: Optional request ID for tracing
        user_id: Optional user ID for context
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = func.__name__
            
            logger.debug(f"Calling {func_name}", extra={
                'request_id': request_id,
                'user_id': user_id,
                'function': func_name,
                'details': {'args': str(args), 'kwargs': str(kwargs)}
            })
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.debug(f"{func_name} completed successfully", extra={
                    'request_id': request_id,
                    'user_id': user_id,
                    'function': func_name,
                    'duration_ms': round(duration_ms, 2)
                })
                
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(f"{func_name} failed", extra={
                    'request_id': request_id,
                    'user_id': user_id,
                    'function': func_name,
                    'duration_ms': round(duration_ms, 2)
                }, exc_info=True)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = func.__name__
            
            logger.debug(f"Calling {func_name}", extra={
                'request_id': request_id,
                'user_id': user_id,
                'function': func_name,
                'details': {'args': str(args), 'kwargs': str(kwargs)}
            })
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.debug(f"{func_name} completed successfully", extra={
                    'request_id': request_id,
                    'user_id': user_id,
                    'function': func_name,
                    'duration_ms': round(duration_ms, 2)
                })
                
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(f"{func_name} failed", extra={
                    'request_id': request_id,
                    'user_id': user_id,
                    'function': func_name,
                    'duration_ms': round(duration_ms, 2)
                }, exc_info=True)
                raise
        
        # Return async wrapper for async functions, sync wrapper for sync functions
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def print_debug(event):
    """Debug mode: formatted output with all event information"""
    from langchain_core.messages import HumanMessage, AIMessage

    print("\n" + "─" * 60)
    for node_name, node_data in event.items():
        print(f"📦 Node: {node_name}")
        print("─" * 60)

        if 'messages' in node_data:
            for msg in node_data['messages']:
                msg_type = type(msg).__name__

                if isinstance(msg, HumanMessage):
                    print(f"👤 HumanMessage:")
                    print(f"   Content: {msg.content}")
                    if hasattr(msg, 'id'):
                        print(f"   ID: {msg.id}")

                elif isinstance(msg, AIMessage):
                    print(f"🤖 AIMessage:")

                    # Content
                    if isinstance(msg.content, str):
                        print(f"   Content: {msg.content}")
                    elif isinstance(msg.content, list):
                        print(f"   Content (list):")
                        for idx, item in enumerate(msg.content):
                            if isinstance(item, dict):
                                if item.get('type') == 'tool_use':
                                    print(f"      [{idx}] Tool Use:")
                                    print(f"         Name: {item.get('name')}")
                                    print(f"         ID: {item.get('id')}")
                                    print(f"         Input: {json.dumps(item.get('input'), ensure_ascii=False)}")
                                elif item.get('type') == 'text':
                                    print(f"      [{idx}] Text: {item.get('text')}")
                                else:
                                    print(f"      [{idx}] {json.dumps(item, ensure_ascii=False)}")
                            else:
                                print(f"      [{idx}] {item}")

                    # Additional kwargs
                    if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                        print(f"   Additional kwargs: {json.dumps(msg.additional_kwargs, ensure_ascii=False)}")

                    # Tool calls
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print(f"   Tool calls:")
                        for tc in msg.tool_calls:
                            print(f"      - Name: {tc.get('name')}")
                            print(f"        ID: {tc.get('id')}")
                            print(f"        Args: {json.dumps(tc.get('args'), ensure_ascii=False)}")
                            print(f"        Type: {tc.get('type')}")

                    # Response metadata
                    if hasattr(msg, 'response_metadata') and msg.response_metadata:
                        print(f"   Response metadata:")
                        metadata = msg.response_metadata
                        print(f"      ID: {metadata.get('id')}")
                        print(f"      Model: {metadata.get('model_name')}")
                        print(f"      Provider: {metadata.get('model_provider')}")
                        print(f"      Stop reason: {metadata.get('stop_reason')}")
                        if 'usage' in metadata:
                            print(f"      Usage:")
                            usage = metadata['usage']
                            for k, v in usage.items():
                                print(f"         {k}: {v}")

                    # Usage metadata
                    if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                        print(f"   Usage metadata: {json.dumps(msg.usage_metadata, ensure_ascii=False)}")

                    # ID
                    if hasattr(msg, 'id'):
                        print(f"   Message ID: {msg.id}")

                else:
                    # Tool message or other types
                    print(f"🔧 {msg_type}:")
                    if hasattr(msg, 'content'):
                        print(f"   Content: {msg.content}")
                    if hasattr(msg, 'name'):
                        print(f"   Tool name: {msg.name}")
                    if hasattr(msg, 'tool_call_id'):
                        print(f"   Tool call ID: {msg.tool_call_id}")
                    if hasattr(msg, 'id'):
                        print(f"   Message ID: {msg.id}")

                print()
        else:
            # Non-message data
            print(json.dumps(node_data, indent=2, ensure_ascii=False, default=str))
            print()


def print_simple(event):
    """Simple mode: only output the last AI message"""
    from langchain_core.messages import AIMessage

    for key, value in event.items():
        if 'messages' in value and len(value['messages']) > 0:
            last_msg = value['messages'][-1]
            if isinstance(last_msg, AIMessage):
                print(last_msg.content)
