"""
Main OCR processing client orchestrator.

Coordinates document fetching, OCR submission, and result storage.
Optimized for maximum throughput with minimal idle time.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

from .config import ClientConfig
from .database import OCRDatabasePool, DocumentOperations
from .path_builder import build_full_path, to_container_path
from .job_submitter import JobSubmitter
from .result_processor import build_ocr_result, extract_text_for_document, is_zero_text

logger = logging.getLogger(__name__)

# Optimization constants
FAST_POLL_INTERVAL = 0.3  # Fast polling when jobs are pending
IDLE_POLL_INTERVAL = 1.0  # Slower polling when idle

# Resilience constants
JOB_TIMEOUT_SECONDS = 600  # Abandon jobs older than 10 minutes
MAX_NOT_FOUND_ERRORS = 3  # Abandon job after this many "not found" errors


@dataclass
class ProcessingStats:
    """Statistics for the processing run."""
    start_time: float = field(default_factory=time.time)
    documents_fetched: int = 0
    documents_with_files: int = 0
    documents_missing_files: int = 0
    batches_submitted: int = 0
    batches_completed: int = 0
    documents_processed: int = 0
    documents_saved: int = 0
    pages_processed: int = 0
    fallback_attempted: int = 0
    fallback_recovered: int = 0
    errors: int = 0

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def throughput_docs_per_sec(self) -> float:
        elapsed = self.elapsed()
        return self.documents_processed / elapsed if elapsed > 0 else 0

    def throughput_pages_per_sec(self) -> float:
        elapsed = self.elapsed()
        return self.pages_processed / elapsed if elapsed > 0 else 0


@dataclass
class PendingJob:
    """Tracks a submitted job and its documents."""
    job_id: str
    documents: List[Dict]
    container_paths: List[str]
    submit_time: float = field(default_factory=time.time)
    not_found_errors: int = 0  # Track consecutive "not found" errors


class PaddleOCRClient:
    """
    Production client for processing documents through paddle_v3 OCR server.

    Features:
    - Streaming producer-consumer pipeline with keyset pagination
    - Bounded buffer with automatic backpressure
    - Queue-aware batch submission with continuous pipelining
    - Parallel status checks for all pending jobs
    - Background database writes via asyncio.Queue (non-blocking)
    - Fast polling (0.3s) for minimal idle time
    - Flat memory usage regardless of target count
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self.db_pool: Optional[OCRDatabasePool] = None
        self.db_ops: Optional[DocumentOperations] = None
        self.job_submitter: Optional[JobSubmitter] = None
        self.stats = ProcessingStats()
        self._pending_jobs: Dict[str, PendingJob] = {}
        self._shutdown = False
        self._db_save_queue: asyncio.Queue[List[Dict]] = asyncio.Queue()
        self._db_save_task: Optional[asyncio.Task] = None
        self._doc_buffer: Optional[asyncio.Queue] = None

    async def initialize(self):
        """Initialize database and HTTP connections."""
        # Initialize database pool
        self.db_pool = OCRDatabasePool(
            supabase_url=self.config.supabase_url,
            database_password=self.config.database_password,
            max_connections=self.config.db_max_connections,
        )
        await self.db_pool.initialize()
        self.db_ops = DocumentOperations(self.db_pool)

        # Initialize HTTP client
        self.job_submitter = JobSubmitter(
            server_url=self.config.ocr_server_url,
            timeout=300.0,
        )

        # Verify server is healthy
        if not await self.job_submitter.health_check():
            raise RuntimeError(
                f"OCR server at {self.config.ocr_server_url} is not healthy"
            )

        logger.info(f"Client initialized, connected to {self.config.ocr_server_url}")

        # Start background DB save worker
        self._db_save_task = asyncio.create_task(self._db_save_worker())

    async def close(self):
        """Close all connections."""
        self._shutdown = True

        # Wait for pending DB saves
        if self._db_save_task:
            if not self._db_save_queue.empty():
                await self._db_save_queue.join()
            self._db_save_task.cancel()
            try:
                await self._db_save_task
            except asyncio.CancelledError:
                pass

        if self.job_submitter:
            await self.job_submitter.close()

        if self.db_pool:
            await self.db_pool.close()

        logger.info("Client closed")

    async def _db_save_worker(self):
        """Background worker for non-blocking database saves."""
        while True:
            try:
                results = await asyncio.wait_for(
                    self._db_save_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                if self._shutdown:
                    break
                continue
            except asyncio.CancelledError:
                break

            try:
                saved = await self.db_ops.save_ocr_results(results)
                self.stats.documents_saved += saved
            except Exception as e:
                logger.error(f"Background DB save failed: {e}")
                self.stats.errors += 1
            finally:
                self._db_save_queue.task_done()

    async def run(self, target_count: Optional[int] = None) -> ProcessingStats:
        """
        Main processing loop using streaming producer-consumer pipeline.

        Documents are fetched in chunks (keyset pagination), validated, and
        pushed into a bounded buffer. A consumer reads from the buffer,
        assembles batches, and submits them to the OCR server. Backpressure
        is automatic: if OCR is slow, the producer pauses when the buffer
        is full. Memory stays flat regardless of target count.

        Args:
            target_count: Stop after processing this many documents (None = unlimited)

        Returns:
            Final processing statistics
        """
        target = target_count or self.config.target_count
        self.stats = ProcessingStats()

        logger.info(f"Starting processing run (mode={self.config.mode}, target={target or 'unlimited'})")

        try:
            # Show available count before fetching (non-fatal if it times out)
            try:
                available = await self.db_ops.get_document_count(
                    mode=self.config.mode,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    excluded_types=self.config.excluded_doc_types,
                    included_types=self.config.included_doc_types,
                    uploaded_after=self.config.uploaded_after,
                    uploaded_before=self.config.uploaded_before,
                    reprocess_before=self.config.reprocess_before,
                )
                approx = "~" if self.config.mode == "new" and not any([
                    self.config.start_date, self.config.end_date,
                    self.config.included_doc_types,
                    self.config.uploaded_after, self.config.uploaded_before,
                ]) else ""
                logger.info(f"Available documents matching criteria: {approx}{available:,}")
                if target:
                    logger.info(f"Targeting: {min(target, available):,} documents")
            except Exception as e:
                logger.warning(f"Count query timed out ({e}), proceeding without count")
                available = None

            # Create bounded buffer for producer-consumer pipeline
            buffer_size = self.config.batch_size * 8
            self._doc_buffer = asyncio.Queue(maxsize=buffer_size)

            # Start producer (fetches chunks from DB, validates files, pushes to buffer)
            producer = asyncio.create_task(self._document_producer(target))

            # Start stats printer
            stats_task = asyncio.create_task(self._stats_printer())

            try:
                # Consume documents and process in batches
                await self._process_from_buffer()
            finally:
                stats_task.cancel()
                try:
                    await stats_task
                except asyncio.CancelledError:
                    pass

            # Wait for producer to finish (should already be done)
            await producer

            # Wait for any remaining jobs
            await self._wait_for_pending_jobs()

        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.stats.errors += 1
            raise

        self._print_final_stats()
        return self.stats

    async def _document_producer(self, target: Optional[int]):
        """
        Fetch documents in chunks via keyset pagination, validate files,
        and push valid documents into the bounded buffer.

        Backpressure: blocks on put() when the buffer is full, which
        naturally throttles fetching to match OCR throughput.
        """
        cursor_recorded_at = None
        cursor_id = None
        produced = 0
        chunk_size = 500

        try:
            while not self._shutdown:
                # Check if we've produced enough
                if target and produced >= target:
                    break

                # Adjust chunk size for final chunk
                fetch_size = chunk_size
                if target:
                    fetch_size = min(fetch_size, target - produced)

                docs, next_cursor_ra, next_cursor_id = await self.db_ops.fetch_documents_chunked(
                    mode=self.config.mode,
                    chunk_size=fetch_size,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    excluded_types=self.config.excluded_doc_types,
                    included_types=self.config.included_doc_types,
                    uploaded_after=self.config.uploaded_after,
                    uploaded_before=self.config.uploaded_before,
                    reprocess_before=self.config.reprocess_before,
                    cursor_recorded_at=cursor_recorded_at,
                    cursor_id=cursor_id,
                )

                if not docs:
                    break

                self.stats.documents_fetched += len(docs)

                # Validate files and push to buffer
                for doc in docs:
                    if self._shutdown:
                        break
                    if target and produced >= target:
                        break

                    full_path = build_full_path(doc, self.config.host_pdf_path)
                    if full_path:
                        doc["_host_path"] = full_path
                        doc["_container_path"] = to_container_path(
                            full_path,
                            self.config.host_pdf_path,
                            self.config.container_pdf_path,
                        )
                        # Blocks if buffer is full (backpressure)
                        await self._doc_buffer.put(doc)
                        self.stats.documents_with_files += 1
                        produced += 1
                    else:
                        self.stats.documents_missing_files += 1
                        logger.debug(f"Missing file for document {doc.get('id')}")

                # Advance cursor
                cursor_recorded_at = next_cursor_ra
                cursor_id = next_cursor_id

                if cursor_recorded_at is None:
                    break  # No more documents

                logger.debug(
                    f"Fetched chunk: {len(docs)} docs, "
                    f"{produced} valid so far, "
                    f"{self.stats.documents_missing_files} missing"
                )

        except Exception as e:
            logger.error(f"Document producer error: {e}")
            self.stats.errors += 1
        finally:
            # Signal end of stream
            await self._doc_buffer.put(None)

        logger.info(
            f"Producer finished: {self.stats.documents_fetched} fetched, "
            f"{produced} valid, {self.stats.documents_missing_files} missing files"
        )

    async def _process_from_buffer(self):
        """
        Read documents from the buffer, assemble batches, and submit to OCR.

        Periodically checks for completed jobs while waiting for documents.
        """
        batch: List[Dict] = []

        while not self._shutdown:
            try:
                # Get next document (with timeout to allow checking completed jobs)
                doc = await asyncio.wait_for(self._doc_buffer.get(), timeout=0.5)
            except asyncio.TimeoutError:
                # Check for completed jobs while waiting
                await self._check_completed_jobs()
                continue

            if doc is None:
                # End of stream - submit any remaining partial batch
                if batch:
                    await self._wait_for_capacity()
                    await self._submit_batch(batch)
                break

            batch.append(doc)

            if len(batch) >= self.config.batch_size:
                await self._wait_for_capacity()
                await self._submit_batch(batch)
                batch = []

                # Check for completed jobs after each batch submission
                await self._check_completed_jobs()

    async def _wait_for_capacity(self):
        """Wait until we have capacity for more jobs."""
        while len(self._pending_jobs) >= self.config.max_concurrent_batches:
            if self._shutdown:
                return
            await self._check_completed_jobs()
            if len(self._pending_jobs) >= self.config.max_concurrent_batches:
                await asyncio.sleep(FAST_POLL_INTERVAL)  # Fast polling

    async def _submit_batch(self, documents: List[Dict]):
        """Submit a batch of documents for OCR."""
        await self._submit_single_batch(documents, dpi=None)

    async def _submit_single_batch(
        self, documents: List[Dict], dpi: Optional[int] = None
    ):
        """Submit a single batch of documents for OCR."""
        container_paths = [doc["_container_path"] for doc in documents]

        try:
            job_id = await self.job_submitter.submit_batch(container_paths, dpi=dpi)

            self._pending_jobs[job_id] = PendingJob(
                job_id=job_id,
                documents=documents,
                container_paths=container_paths,
            )
            self.stats.batches_submitted += 1

        except Exception as e:
            logger.error(f"Failed to submit batch: {e}")
            self.stats.errors += 1

    async def _check_completed_jobs(self):
        """Check for completed jobs and process results (parallel status checks)."""
        if not self._pending_jobs:
            return

        # Check all jobs in parallel for speed
        async def check_one(job_id: str):
            try:
                status = await self.job_submitter.get_status(job_id)
                return job_id, status.get("status", "unknown"), status, None
            except Exception as e:
                return job_id, "error", {}, e

        results = await asyncio.gather(
            *[check_one(job_id) for job_id in self._pending_jobs.keys()]
        )

        completed = []
        abandoned = []
        current_time = time.time()

        for job_id, job_status, status_data, error in results:
            pending = self._pending_jobs.get(job_id)
            if not pending:
                continue

            if error:
                error_str = str(error)
                # Check for "Job not found" - indicates server restart
                if "not found" in error_str.lower() or "404" in error_str:
                    pending.not_found_errors += 1
                    if pending.not_found_errors >= MAX_NOT_FOUND_ERRORS:
                        logger.warning(
                            f"Abandoning stale job {job_id} after {pending.not_found_errors} "
                            f"'not found' errors (server likely restarted)"
                        )
                        abandoned.append(job_id)
                else:
                    logger.error(f"Error checking job {job_id}: {error}")
                continue

            # Reset error counter on successful status check
            pending.not_found_errors = 0

            # Check for job timeout
            job_age = current_time - pending.submit_time
            if job_age > JOB_TIMEOUT_SECONDS:
                logger.warning(
                    f"Abandoning timed out job {job_id} (age: {job_age:.0f}s > {JOB_TIMEOUT_SECONDS}s)"
                )
                abandoned.append(job_id)
                continue

            if job_status in ("completed", "completed_with_errors", "partial"):
                completed.append(job_id)
            elif job_status == "failed":
                logger.error(f"Job {job_id} failed")
                completed.append(job_id)
                self.stats.errors += 1
            elif job_status == "processing":
                # Detect stuck jobs where all pages are done but status
                # didn't transition (e.g. due to missing/dropped pages)
                total = status_data.get("total_pages", 0)
                processed = status_data.get("processed_pages", 0)
                dropped = status_data.get("dropped_pages", 0)
                if total > 0 and processed + dropped >= total:
                    logger.warning(
                        f"Job {job_id} stuck in 'processing' with all pages done "
                        f"({processed} processed, {dropped} dropped) — treating as completed"
                    )
                    completed.append(job_id)

        # Remove abandoned jobs (re-queue their documents for next run)
        for job_id in abandoned:
            pending = self._pending_jobs.pop(job_id, None)
            if pending:
                self.stats.errors += 1
                logger.info(f"Abandoned job {job_id} with {len(pending.documents)} documents")

        # Process completed jobs (can also parallelize if needed)
        for job_id in completed:
            await self._process_completed_job(job_id)

    async def _process_completed_job(self, job_id: str):
        """Process a completed job and save results.

        After building OCR results, detects zero-text documents and
        resubmits them at higher DPI to the same server as a fallback.
        """
        pending = self._pending_jobs.pop(job_id, None)
        if not pending:
            return

        try:
            # Check for missing pages before fetching results
            try:
                status = await self.job_submitter.get_status(job_id)
                missing_pages = status.get("missing_pages", {})
                if missing_pages:
                    total_missing = sum(len(p) for p in missing_pages.values())
                    logger.warning(
                        f"Job {job_id}: {total_missing} missing pages across "
                        f"{len(missing_pages)} documents"
                    )
                    for pdf_path, pages in missing_pages.items():
                        logger.debug(f"  Missing pages in {pdf_path}: {pages}")
            except Exception:
                pass

            # Get results
            job_results = await self.job_submitter.get_results(job_id)
            processing_time = time.time() - pending.submit_time

            # Build OCR results for each document
            ocr_results = []
            fallback_docs = []  # (doc, container_path) pairs with zero text
            results_dict = job_results.get("results", {})

            for doc, container_path in zip(pending.documents, pending.container_paths):
                # Find this document's results in the job
                doc_results = self._extract_doc_results(results_dict, container_path, doc.get('id'))

                if doc_results:
                    # Build individual job result structure
                    single_doc_results = {
                        "job_id": job_id,
                        "status": job_results.get("status"),
                        "total_pages": sum(len(pages) for pages in doc_results.values()),
                        "results": doc_results,
                    }

                    ocr_result = build_ocr_result(doc, single_doc_results, processing_time)

                    if is_zero_text(ocr_result):
                        fallback_docs.append((doc, container_path))
                    else:
                        ocr_results.append(ocr_result)
                        self.stats.pages_processed += single_doc_results["total_pages"]

            # Fallback: resubmit zero-text docs at higher DPI
            if fallback_docs:
                self.stats.fallback_attempted += len(fallback_docs)
                fallback_results = await self._run_fallback(
                    fallback_docs, processing_time
                )
                ocr_results.extend(fallback_results)

            # Queue for background save (non-blocking)
            if ocr_results:
                await self._db_save_queue.put(ocr_results)
                self.stats.documents_processed += len(ocr_results)

            self.stats.batches_completed += 1

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            self.stats.errors += 1

    async def _run_fallback(
        self,
        fallback_docs: List[tuple],
        original_processing_time: float,
    ) -> List[Dict]:
        """Resubmit zero-text documents at higher DPI and return OCR results.

        Args:
            fallback_docs: List of (doc, container_path) tuples
            original_processing_time: Time from original submission

        Returns:
            List of OCR result dicts (with fallback metadata)
        """
        fallback_dpi = self.config.fallback_dpi
        container_paths = [cp for _, cp in fallback_docs]

        logger.info(
            f"Fallback: resubmitting {len(fallback_docs)} zero-text docs at {fallback_dpi} DPI"
        )

        try:
            fb_job_id = await self.job_submitter.submit_batch(
                container_paths, dpi=fallback_dpi
            )
            fb_results = await self.job_submitter.wait_for_completion(
                fb_job_id, poll_interval=FAST_POLL_INTERVAL
            )
        except Exception as e:
            logger.error(f"Fallback submission failed: {e}")
            self.stats.errors += 1
            return []

        fb_results_dict = fb_results.get("results", {})
        recovered = []

        for doc, container_path in fallback_docs:
            doc_results = self._extract_doc_results(
                fb_results_dict, container_path, doc.get("id")
            )
            if doc_results:
                single_doc_results = {
                    "job_id": fb_job_id,
                    "status": fb_results.get("status"),
                    "total_pages": sum(
                        len(pages) for pages in doc_results.values()
                    ),
                    "results": doc_results,
                }

                ocr_result = build_ocr_result(
                    doc, single_doc_results, original_processing_time
                )

                # Tag with fallback metadata
                metadata = json.loads(ocr_result["ocr_metadata"])
                metadata["fallback"] = True
                metadata["fallback_dpi"] = fallback_dpi
                ocr_result["ocr_metadata"] = json.dumps(metadata)

                recovered.append(ocr_result)
                self.stats.pages_processed += single_doc_results["total_pages"]

        # Guard: drop results that are still zero-text after fallback.
        # In reprocess mode this prevents overwriting existing good text
        # with an empty string.
        non_empty = [r for r in recovered if not is_zero_text(r)]
        dropped = len(recovered) - len(non_empty)
        self.stats.fallback_recovered += len(non_empty)

        if dropped:
            dropped_lookup = {doc["id"]: doc for doc, _ in fallback_docs}
            dropped_ids = set(r.get("document_id") for r in recovered) - set(r.get("document_id") for r in non_empty)
            for did in dropped_ids:
                doc = dropped_lookup.get(did, {})
                logger.warning(
                    f"Fallback: dropping zero-text document_id={did} "
                    f"source_document_id={doc.get('source_document_id', '?')}"
                )
            logger.warning(
                f"Fallback: dropped {dropped} still-empty results to avoid "
                f"overwriting existing data"
            )

        logger.info(
            f"Fallback: {len(non_empty)}/{len(fallback_docs)} docs recovered at {fallback_dpi} DPI"
        )

        return non_empty

    def _extract_doc_results(
        self,
        results_dict: Dict[str, Dict],
        container_path: str,
        document_id: str = None,
    ) -> Optional[Dict[str, Dict]]:
        """Extract results for a specific document from job results."""
        # Direct match
        if container_path in results_dict:
            return {container_path: results_dict[container_path]}

        # Try matching by filename
        target_filename = container_path.split("/")[-1]
        for path, pages in results_dict.items():
            if path.endswith(target_filename):
                logger.warning(
                    f"Path mismatch for doc {document_id}: "
                    f"expected '{container_path}', matched '{path}'"
                )
                return {path: pages}

        logger.error(
            f"No results found for doc {document_id}: "
            f"path '{container_path}' not in {list(results_dict.keys())}"
        )
        return None

    async def _wait_for_pending_jobs(self):
        """Wait for all pending jobs to complete."""
        last_log = 0
        while self._pending_jobs:
            now = time.time()
            if now - last_log >= 2.0:  # Log every 2 seconds
                logger.info(f"Waiting for {len(self._pending_jobs)} pending jobs...")
                last_log = now
            await self._check_completed_jobs()
            if self._pending_jobs:
                await asyncio.sleep(FAST_POLL_INTERVAL)

        # Wait for background DB saves to complete
        if not self._db_save_queue.empty():
            logger.info(f"Waiting for ~{self._db_save_queue.qsize()} pending DB saves...")
            await self._db_save_queue.join()

    async def _stats_printer(self):
        """Periodically print processing statistics."""
        while True:
            await asyncio.sleep(self.config.stats_print_interval)
            self._print_progress()

    def _print_progress(self):
        """Print current progress."""
        elapsed = self.stats.elapsed()
        docs_per_sec = self.stats.throughput_docs_per_sec()
        pages_per_sec = self.stats.throughput_pages_per_sec()

        logger.info(
            f"Progress: {self.stats.documents_processed} docs, "
            f"{self.stats.pages_processed} pages | "
            f"{docs_per_sec:.2f} docs/s, {pages_per_sec:.2f} pages/s | "
            f"Pending: {len(self._pending_jobs)} jobs | "
            f"Elapsed: {elapsed:.0f}s"
        )

    def _print_final_stats(self):
        """Print final statistics."""
        elapsed = self.stats.elapsed()
        docs_per_sec = self.stats.throughput_docs_per_sec()
        pages_per_sec = self.stats.throughput_pages_per_sec()

        logger.info("=" * 60)
        logger.info("Processing Complete")
        logger.info("=" * 60)
        logger.info(f"Documents fetched:      {self.stats.documents_fetched}")
        logger.info(f"Documents with files:   {self.stats.documents_with_files}")
        logger.info(f"Documents missing:      {self.stats.documents_missing_files}")
        logger.info(f"Documents processed:    {self.stats.documents_processed}")
        logger.info(f"Documents saved:        {self.stats.documents_saved}")
        logger.info(f"Pages processed:        {self.stats.pages_processed}")
        logger.info(f"Fallback attempted:     {self.stats.fallback_attempted}")
        logger.info(f"Fallback recovered:     {self.stats.fallback_recovered}")
        logger.info(f"Errors:                 {self.stats.errors}")
        logger.info("-" * 60)
        logger.info(f"Elapsed time:           {elapsed:.1f}s")
        logger.info(f"Throughput (docs):      {docs_per_sec:.2f} docs/s")
        logger.info(f"Throughput (pages):     {pages_per_sec:.2f} pages/s")
        logger.info("=" * 60)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
