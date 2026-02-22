"""Client configuration with environment variable support."""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ClientConfig:
    """Configuration for the OCR processing client."""

    # Database Configuration
    supabase_url: str = ""
    supabase_key: str = ""
    database_password: Optional[str] = None
    db_host: str = "aws-0-us-east-2.pooler.supabase.com"
    db_port: int = 5432
    db_name: str = "postgres"
    db_user: str = "postgres.jufpdohfvviyyxmpqebj"
    db_max_connections: int = 6

    # OCR Server Configuration
    ocr_server_url: str = "http://localhost:8003"

    # Processing Mode
    mode: str = "new"  # "new" for unprocessed, "reprocess" for existing

    # Batch Configuration (optimized for ~3.0 pages/s throughput)
    batch_size: int = 16  # Documents per OCR batch
    max_concurrent_batches: int = 4  # Max jobs in flight

    # Polling Configuration
    status_poll_interval: float = 2.0  # Seconds between status checks
    stats_print_interval: float = 30.0  # Seconds between progress prints

    # Fallback Configuration
    fallback_dpi: int = 300  # DPI for zero-text retry

    # Path Configuration
    host_pdf_path: str = "/mnt/models/wake-county-pdfs"
    container_pdf_path: str = "/data"

    # Filtering (optional)
    start_date: Optional[str] = None  # YYYY-MM-DD format
    end_date: Optional[str] = None
    target_count: Optional[int] = None  # Stop after N documents

    # Document type filtering
    excluded_doc_types: list = field(default_factory=lambda: [
        "UNUSED FILE NUMBER",
        "BLANK",
    ])
    included_doc_types: Optional[List[str]] = None  # Only process these types

    # Upload date filtering (created_at in database)
    uploaded_after: Optional[str] = None  # YYYY-MM-DD format
    uploaded_before: Optional[str] = None  # YYYY-MM-DD format

    # Reprocess filtering (ocr_results.updated_at)
    reprocess_before: Optional[str] = None  # Only reprocess docs with OCR updated before this date

    @classmethod
    def from_env(cls) -> "ClientConfig":
        """Create config from environment variables."""
        return cls(
            # Database
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_key=os.getenv("SUPABASE_KEY", ""),
            database_password=os.getenv("DATABASE_PASSWORD"),
            db_host=os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
            db_port=int(os.getenv("DB_PORT", "5432")),
            db_name=os.getenv("DB_NAME", "postgres"),
            db_user=os.getenv("DB_USER", "postgres.jufpdohfvviyyxmpqebj"),
            db_max_connections=int(os.getenv("DB_MAX_CONNECTIONS", "6")),

            # OCR Server
            ocr_server_url=os.getenv("OCR_SERVER_URL", "http://localhost:8003"),

            # Processing
            mode=os.getenv("MODE", "new"),
            batch_size=int(os.getenv("BATCH_SIZE", "16")),
            max_concurrent_batches=int(os.getenv("MAX_CONCURRENT_BATCHES", "3")),
            status_poll_interval=float(os.getenv("STATUS_POLL_INTERVAL", "2.0")),
            stats_print_interval=float(os.getenv("STATS_PRINT_INTERVAL", "30.0")),

            # Paths
            host_pdf_path=os.getenv("HOST_PDF_PATH", "/mnt/models/wake-county-pdfs"),
            container_pdf_path=os.getenv("CONTAINER_PDF_PATH", "/data"),

            # Filtering
            start_date=os.getenv("START_DATE"),
            end_date=os.getenv("END_DATE"),
            target_count=int(os.getenv("TARGET_COUNT")) if os.getenv("TARGET_COUNT") else None,
            included_doc_types=os.getenv("INCLUDED_DOC_TYPES", "").split(",") if os.getenv("INCLUDED_DOC_TYPES") else None,
            uploaded_after=os.getenv("UPLOADED_AFTER"),
            uploaded_before=os.getenv("UPLOADED_BEFORE"),
        )

    def get_db_dsn(self) -> str:
        """Get database connection string."""
        password = self.database_password or ""
        return f"postgresql://{self.db_user}:{password}@{self.db_host}:{self.db_port}/{self.db_name}"
