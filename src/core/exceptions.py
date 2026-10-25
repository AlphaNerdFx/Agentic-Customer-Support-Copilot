"""Custom exception classes for FDE Copilot."""


class FDECopilotError(Exception):
    """Base exception for all domain errors."""


class IngestionError(FDECopilotError):
    """Raised when knowledge base ingestion fails."""


class RetrievalError(FDECopilotError):
    """Raised when vector retrieval fails."""


class ToolExecutionError(FDECopilotError):
    """Raised when tool execution fails."""


class ApprovalGateError(FDECopilotError):
    """Raised when human-in-the-loop state transition fails."""
