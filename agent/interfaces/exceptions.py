"""Exception Definitions"""


# Memory exceptions
class MemoryError(Exception):
    """Base class for memory-related errors"""
    pass


class MemoryStorageError(MemoryError):
    """Error occurred while storing or retrieving data"""
    pass


class SessionNotFoundError(MemoryError):
    """Requested session does not exist"""
    pass


# Storage exceptions
class StorageError(Exception):
    """Base class for storage backend errors"""
    pass


class ConnectionError(StorageError):
    """Cannot connect to storage backend"""
    pass


class ConfigurationError(StorageError):
    """Storage configuration error"""
    pass


# Extraction exceptions
class ExtractionError(Exception):
    """Error during context extraction process"""
    pass


class ValidationError(Exception):
    """Input or configuration validation error"""
    pass


# Embedding exceptions
class EmbeddingError(Exception):
    """Error generating embedding vectors"""
    pass
