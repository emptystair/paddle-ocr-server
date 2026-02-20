"""
Result extraction and formatting for database storage.

Processes OCR server results into the format needed for ocr_results table.
"""

import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def is_zero_text(ocr_result: Dict) -> bool:
    """Check if an OCR result contains no text.

    Args:
        ocr_result: Dict with at least a 'full_text' key.

    Returns:
        True if the document produced no text.
    """
    return not ocr_result.get("full_text", "").strip()


def extract_page_text(page_result: Dict) -> str:
    """
    Extract text from a single page result.

    Args:
        page_result: Page result dict with text_lines

    Returns:
        Combined text from all lines
    """
    text_lines = page_result.get("text_lines", [])
    if not text_lines:
        return ""

    # Combine all text lines with newlines
    return "\n".join(line.get("text", "") for line in text_lines)


def extract_page_confidence(page_result: Dict) -> float:
    """
    Calculate average confidence for a page.

    Args:
        page_result: Page result dict with text_lines

    Returns:
        Average confidence (0-100 scale) or 0 if no text
    """
    text_lines = page_result.get("text_lines", [])
    if not text_lines:
        return 0.0

    confidences = [line.get("confidence", 0) for line in text_lines]
    if not confidences:
        return 0.0

    avg = sum(confidences) / len(confidences)
    # Confidence is 0-1 scale, convert to percentage
    return avg * 100 if avg <= 1 else avg


def extract_full_text(results: Dict[str, Dict[int, Dict]]) -> str:
    """
    Extract full text from all pages of a document.

    Args:
        results: Results dict keyed by pdf_path -> page_num -> page_result

    Returns:
        Combined text from all pages, separated by double newlines
    """
    all_text = []

    for pdf_path in sorted(results.keys()):
        page_results = results[pdf_path]
        # Sort by page number
        for page_num in sorted(page_results.keys(), key=int):
            page_result = page_results[page_num]
            if page_result.get("status") == "success":
                page_text = extract_page_text(page_result)
                if page_text:
                    all_text.append(page_text)

    return "\n\n".join(all_text)


def build_pages_json(results: Dict[str, Dict[int, Dict]]) -> List[Dict]:
    """
    Build pages array for database storage.

    Args:
        results: Results dict keyed by pdf_path -> page_num -> page_result

    Returns:
        List of page dicts with text, confidence, page_number
    """
    pages = []

    for pdf_path in sorted(results.keys()):
        page_results = results[pdf_path]
        for page_num in sorted(page_results.keys(), key=int):
            page_result = page_results[page_num]

            page_data = {
                "page_number": int(page_num),
                "status": page_result.get("status", "unknown"),
            }

            if page_result.get("status") == "success":
                page_data["text"] = extract_page_text(page_result)
                page_data["confidence"] = extract_page_confidence(page_result)
                page_data["text_lines_count"] = len(page_result.get("text_lines", []))
            elif page_result.get("status") == "error":
                page_data["error"] = page_result.get("error", "Unknown error")

            pages.append(page_data)

    return pages


def calculate_overall_confidence(results: Dict[str, Dict[int, Dict]]) -> float:
    """
    Calculate overall confidence across all pages.

    Args:
        results: Results dict keyed by pdf_path -> page_num -> page_result

    Returns:
        Average confidence (0-100 scale)
    """
    all_confidences = []

    for pdf_path in results.keys():
        page_results = results[pdf_path]
        for page_num, page_result in page_results.items():
            if page_result.get("status") == "success":
                conf = extract_page_confidence(page_result)
                if conf > 0:
                    all_confidences.append(conf)

    if not all_confidences:
        return 0.0

    return sum(all_confidences) / len(all_confidences)


def build_ocr_result(
    document: Dict,
    job_results: Dict,
    processing_time: float,
) -> Dict[str, Any]:
    """
    Build ocr_results record from job results.

    Args:
        document: Document record from database
        job_results: Results from OCR server (from get_results API)
        processing_time: Total processing time in seconds

    Returns:
        Dict ready for database insertion with:
        - document_id
        - full_text
        - pages (JSON)
        - confidence
        - processing_time
        - ocr_metadata
    """
    # Get results dict (keyed by pdf_path -> page_num -> result)
    results = job_results.get("results", {})

    # Extract full text
    full_text = extract_full_text(results)

    # Build pages array
    pages = build_pages_json(results)

    # Calculate overall confidence
    confidence = calculate_overall_confidence(results)

    # Build metadata
    metadata = {
        "engine": "paddle_v4_trt",
        "model": "PP-OCRv5",
        "job_id": job_results.get("job_id"),
        "total_pages": job_results.get("total_pages", len(pages)),
        "job_status": job_results.get("status"),
        "job_process_time": job_results.get("process_time"),
    }

    # Note: pages column is integer (page count), not JSONB
    # ocr_metadata needs json.dumps() for JSONB column
    return {
        "document_id": document["id"],
        "full_text": full_text,
        "pages": len(pages),  # Integer page count
        "confidence": confidence,
        "processing_time": processing_time,
        "ocr_metadata": json.dumps(metadata),  # JSON string for JSONB column
    }


def build_ocr_results_batch(
    documents: List[Dict],
    job_results_list: List[Dict],
    processing_times: List[float],
) -> List[Dict[str, Any]]:
    """
    Build ocr_results records for a batch of documents.

    Args:
        documents: List of document records
        job_results_list: List of job results from OCR server
        processing_times: List of processing times

    Returns:
        List of dicts ready for database insertion
    """
    results = []

    for doc, job_results, proc_time in zip(
        documents, job_results_list, processing_times
    ):
        try:
            result = build_ocr_result(doc, job_results, proc_time)
            results.append(result)
        except Exception as e:
            logger.error(
                f"Failed to build OCR result for document {doc.get('id')}: {e}"
            )

    return results


def extract_text_for_document(
    job_results: Dict,
    pdf_path: str,
) -> Optional[str]:
    """
    Extract text for a specific document from job results.

    Used when a single job contains multiple documents.

    Args:
        job_results: Results from OCR server
        pdf_path: PDF path to extract text for

    Returns:
        Combined text for the document or None if not found
    """
    results = job_results.get("results", {})

    if pdf_path not in results:
        # Try matching by filename
        target_filename = pdf_path.split("/")[-1]
        for path in results.keys():
            if path.endswith(target_filename):
                pdf_path = path
                break
        else:
            return None

    page_results = results[pdf_path]
    all_text = []

    for page_num in sorted(page_results.keys(), key=int):
        page_result = page_results[page_num]
        if page_result.get("status") == "success":
            page_text = extract_page_text(page_result)
            if page_text:
                all_text.append(page_text)

    return "\n\n".join(all_text) if all_text else ""
