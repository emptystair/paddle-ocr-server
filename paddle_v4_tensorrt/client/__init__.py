"""
PaddleOCR v3 Database Client

A production client for processing documents from Supabase database
through the paddle_v3 OCR server.
"""

from .config import ClientConfig
from .client import PaddleOCRClient
from .database import OCRDatabasePool, DocumentOperations
from .job_submitter import JobSubmitter

__all__ = [
    "ClientConfig",
    "PaddleOCRClient",
    "OCRDatabasePool",
    "DocumentOperations",
    "JobSubmitter",
]
