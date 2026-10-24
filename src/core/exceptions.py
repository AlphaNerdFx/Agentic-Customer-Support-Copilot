"""Custom exception classes for FDE Copilot."""


class FDECopilotError(Exception):
    """Base exception for all domain errors."""
    pass


class IngestionError(FDECopilotError):
    """Raised when knowledge base ingestion fails."""
    pass


class RetrievalError(FDECopilotError):
    """Raised when vector retrieval fails."""
    pass


class ToolExecutionError(FDECopilotError):
    """Raised when tool execution fails."""
    pass


class ApprovalGateError(FDECopilotError):
    """Raised when human-in-the-loop state transition fails."""
    pass
