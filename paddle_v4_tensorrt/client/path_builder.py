"""
PDF path construction and resolution.

Builds file paths from database records and handles alternative naming conventions.
Adapted from rapid_paddle/docker/production_adapter.py and file_path_helper.py
"""

import os
import re
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def extract_doc_info_from_url(pdf_url: Optional[str]) -> Tuple[str, str]:
    """
    Extract document ID and PDF name from URL.

    Example URLs:
    - New format: https://rodrecords.wake.gov/web/document-image-pdf/DOC742S1053//019957-01985-1.pdf
    - Old format: ...Type=DOC/DOCC110901178/017744-02346-1.pdf

    Returns: (doc_id, pdf_name)
    """
    if not pdf_url:
        return "unknown", "document"

    # New URL format: /document-image-pdf/{doc_id}//{filename}.pdf
    match = re.search(r'/document-image-pdf/([^/]+)//([^/]+)\.pdf', pdf_url)
    if match:
        return match.group(1), match.group(2)

    # Old URL format with Type parameter: Type=DOC/DOCC{id}/{filename}.pdf
    match = re.search(r'Type=DOC/(DOCC\d+)/([^/]+)\.pdf', pdf_url)
    if match:
        return match.group(1), match.group(2)

    # Alternative: Look for DOCC pattern in path
    match = re.search(r'/(DOCC\d+)/([^/]+)\.pdf', pdf_url)
    if match:
        return match.group(1), match.group(2)

    # Another pattern: /DOC{id}/{filename}.pdf
    match = re.search(r'/(DOC[^/]+)/([^/]+)\.pdf', pdf_url)
    if match:
        return match.group(1), match.group(2)

    # Fallback - extract just the filename
    match = re.search(r'/([^/]+)\.pdf', pdf_url)
    if match:
        return "DOC", match.group(1)

    return "unknown", "document"


def construct_file_path(document: Dict) -> str:
    """
    Construct the relative file path from document record.

    Args:
        document: Dict with 'pdf_url' and 'recorded_at' fields

    Returns:
        Relative path like: 2020/02/12/DOCC110901178_017744-02346-1.pdf
    """
    pdf_url = document.get('pdf_url', '')
    recorded_at = document.get('recorded_at')

    # Extract doc ID and PDF name from URL
    doc_id, pdf_name = extract_doc_info_from_url(pdf_url)

    # Construct filename
    filename = f"{doc_id}_{pdf_name}.pdf"

    # Parse date for directory structure
    if recorded_at:
        try:
            # Handle both string and datetime objects
            if isinstance(recorded_at, str):
                dt = datetime.fromisoformat(recorded_at.replace('Z', '+00:00'))
            else:
                dt = recorded_at

            # Create date-based path: YYYY/MM/DD/filename
            date_path = f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d}"
            return f"{date_path}/{filename}"
        except (ValueError, AttributeError, TypeError) as e:
            logger.warning(f"Failed to parse recorded_at '{recorded_at}': {e}")

    # Fallback if no recorded_at or parsing failed
    return f"unknown/{filename}"


def resolve_pdf_path(base_path: str, constructed_path: str) -> str:
    """
    Resolve the actual PDF path by checking multiple naming conventions.

    Args:
        base_path: Base directory like /mnt/models/wake-county-pdfs
        constructed_path: Path constructed from database like 2020/02/12/DOCC123_019511-00182-1.pdf

    Returns:
        The actual file path if found, otherwise the original constructed path
    """
    full_path = f"{base_path}/{constructed_path}"

    # First check if the constructed path exists as-is
    if os.path.exists(full_path):
        return constructed_path

    # Check for double .pdf.pdf extension (common issue)
    if constructed_path.endswith('.pdf'):
        double_pdf_path = f"{constructed_path}.pdf"
        if os.path.exists(f"{base_path}/{double_pdf_path}"):
            logger.debug(f"Found file with double .pdf extension: {double_pdf_path}")
            return double_pdf_path

    # Parse the components and try alternative formats
    dir_path = os.path.dirname(constructed_path)
    filename = os.path.basename(constructed_path)

    # Try to extract pattern: {doc_id}_{book}-{page}-{suffix}.pdf
    match = re.match(r'(DOC[^_]+)_(\d+)-(\d+)-(\d+)\.pdf', filename)
    if match:
        doc_id = match.group(1)
        book = match.group(2)
        page = match.group(3)
        suffix = match.group(4)

        alternative_formats = [
            f"{doc_id}_B: {book} P: {page}-{suffix}.pdf",
            f"{doc_id}_B{book}P{page}-{suffix}.pdf",
            f"{doc_id}_B_{book}_P_{page}-{suffix}.pdf",
        ]

        for alt_filename in alternative_formats:
            alt_path = f"{dir_path}/{alt_filename}"
            if os.path.exists(f"{base_path}/{alt_path}"):
                logger.debug(f"Found alternative path: {alt_path}")
                return alt_path

    return constructed_path


def build_full_path(document: Dict, base_path: str) -> Optional[str]:
    """
    Build and validate the full file path for a document.

    Args:
        document: Document record from database
        base_path: Base directory for PDFs

    Returns:
        Full file path if file exists, None otherwise
    """
    # Construct relative path
    rel_path = construct_file_path(document)

    # Try to resolve (checks alternative naming)
    resolved_path = resolve_pdf_path(base_path, rel_path)

    # Build full path
    full_path = f"{base_path}/{resolved_path}"

    # Return if exists
    if os.path.exists(full_path):
        return full_path

    return None


def to_container_path(host_path: str, host_prefix: str, container_prefix: str) -> str:
    """
    Transform host path to container path.

    Args:
        host_path: Full path on host (e.g., /mnt/models/wake-county-pdfs/2024/01/02/doc.pdf)
        host_prefix: Host base path (e.g., /mnt/models/wake-county-pdfs)
        container_prefix: Container base path (e.g., /data)

    Returns:
        Container path (e.g., /data/2024/01/02/doc.pdf)
    """
    if host_path.startswith(host_prefix):
        relative = host_path[len(host_prefix):]
        return f"{container_prefix}{relative}"
    return host_path
