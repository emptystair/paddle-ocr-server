#!/usr/bin/env python3
"""
CLI entry point for PaddleOCR v4 TensorRT database client.

Usage:
    # Process new documents
    python -m paddle_v4_tensorrt.client.main --mode new --target 10000

    # Reprocess existing documents
    python -m paddle_v4_tensorrt.client.main --mode reprocess --start-date 2024-01-01

    # Quick test
    python -m paddle_v4_tensorrt.client.main --mode new --target 10
"""

import argparse
import asyncio
from datetime import date
import logging
import os
import sys

from dotenv import load_dotenv

from .config import ClientConfig
from .client import PaddleOCRClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PaddleOCR v4 TensorRT Database Client - Process documents from Supabase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 10 new documents (test run)
  python -m paddle_v4_tensorrt.client.main --mode new --target 10

  # Process all unprocessed documents
  python -m paddle_v4_tensorrt.client.main --mode new

  # Reprocess documents from a date range
  python -m paddle_v4_tensorrt.client.main --mode reprocess --start-date 2024-01-01 --end-date 2024-03-01

  # Process all documents recorded in 2025
  python -m paddle_v4_tensorrt.client.main --year 2025

  # Process only documents uploaded today
  python -m paddle_v4_tensorrt.client.main --today --target 100

  # Process only DEED documents
  python -m paddle_v4_tensorrt.client.main --document-type DEED "DEED OF TRUST"

  # Use custom server URL
  python -m paddle_v4_tensorrt.client.main --server http://localhost:8000 --target 100
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["new", "reprocess"],
        default="new",
        help="Processing mode: 'new' for unprocessed docs, 'reprocess' for existing (default: new)",
    )

    # Target count
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Stop after processing N documents (default: unlimited)",
    )

    # Date filtering (recorded_at)
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Filter documents recorded on or after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Filter documents recorded on or before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Shorthand for --start-date YYYY-01-01 --end-date YYYY-12-31",
    )

    # Upload date filtering (created_at)
    parser.add_argument(
        "--uploaded-after",
        type=str,
        default=None,
        help="Filter documents uploaded on or after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--uploaded-before",
        type=str,
        default=None,
        help="Filter documents uploaded on or before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="Shorthand: only documents uploaded today",
    )

    # Reprocess filtering
    parser.add_argument(
        "--reprocess-before",
        type=str,
        default=None,
        help="Only reprocess docs with OCR updated before this date (YYYY-MM-DD). Reprocess mode only.",
    )

    # Document type filtering
    parser.add_argument(
        "--document-type",
        nargs="+",
        default=None,
        metavar="TYPE",
        help='Include only these document types (e.g., --document-type DEED "DEED OF TRUST")',
    )

    # Server configuration
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="OCR server URL (default: http://localhost:8003 or OCR_SERVER_URL env)",
    )

    # Batch configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Documents per OCR batch (default: 16)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum concurrent batches (default: 3)",
    )

    # Path configuration
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="Host path to PDF files (default: /mnt/models/wake-county-pdfs)",
    )
    parser.add_argument(
        "--container-path",
        type=str,
        default=None,
        help="Container path mapping (default: /data)",
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (WARNING level only)",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ClientConfig:
    """Build configuration from environment and CLI args."""
    # Load .env file if present
    load_dotenv()

    # Start with environment-based config
    config = ClientConfig.from_env()

    # Override with CLI arguments
    if args.mode:
        config.mode = args.mode
    if args.target is not None:
        config.target_count = args.target
    if args.start_date:
        config.start_date = args.start_date
    if args.end_date:
        config.end_date = args.end_date
    if args.server:
        config.ocr_server_url = args.server
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_concurrent is not None:
        config.max_concurrent_batches = args.max_concurrent
    if args.pdf_path:
        config.host_pdf_path = args.pdf_path
    if args.container_path:
        config.container_pdf_path = args.container_path

    # --year shorthand: sets start_date and end_date
    if args.year:
        config.start_date = f"{args.year}-01-01"
        config.end_date = f"{args.year}-12-31"

    # Upload date filtering
    if args.uploaded_after:
        config.uploaded_after = args.uploaded_after
    if args.uploaded_before:
        config.uploaded_before = args.uploaded_before

    # --today shorthand: sets uploaded_after and uploaded_before to today
    if args.today:
        today_str = date.today().isoformat()
        config.uploaded_after = today_str
        config.uploaded_before = today_str

    # Reprocess filtering
    if args.reprocess_before:
        config.reprocess_before = args.reprocess_before

    # Document type inclusion filter
    if args.document_type:
        config.included_doc_types = args.document_type

    return config


def validate_config(config: ClientConfig) -> bool:
    """Validate configuration has required values."""
    errors = []

    if not config.supabase_url:
        errors.append("SUPABASE_URL environment variable not set")
    if not config.database_password:
        errors.append("DATABASE_PASSWORD environment variable not set")

    if errors:
        for error in errors:
            logger.error(error)
        return False

    return True


async def main():
    """Main entry point."""
    args = parse_args()

    # Configure log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Build configuration
    config = build_config(args)

    # Validate
    if not validate_config(config):
        logger.error("Configuration validation failed. Check environment variables.")
        sys.exit(1)

    # Log configuration summary
    logger.info("=" * 60)
    logger.info("PaddleOCR v4 TensorRT Database Client")
    logger.info("=" * 60)
    logger.info(f"Mode:           {config.mode}")
    logger.info(f"Target count:   {config.target_count or 'unlimited'}")
    logger.info(f"Server:         {config.ocr_server_url}")
    logger.info(f"Batch size:     {config.batch_size}")
    logger.info(f"Max concurrent: {config.max_concurrent_batches}")
    logger.info(f"PDF path:       {config.host_pdf_path}")
    if config.start_date:
        logger.info(f"Start date:     {config.start_date}")
    if config.end_date:
        logger.info(f"End date:       {config.end_date}")
    if config.uploaded_after:
        logger.info(f"Uploaded after: {config.uploaded_after}")
    if config.uploaded_before:
        logger.info(f"Uploaded before:{config.uploaded_before}")
    if config.reprocess_before:
        logger.info(f"Reprocess before:{config.reprocess_before}")
    if config.included_doc_types:
        logger.info(f"Doc types:      {', '.join(config.included_doc_types)}")
    logger.info("=" * 60)

    # Run client
    try:
        async with PaddleOCRClient(config) as client:
            stats = await client.run()

            # Exit with error if there were errors
            if stats.errors > 0:
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
