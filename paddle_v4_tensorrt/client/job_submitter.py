"""
OCR Server HTTP client for job submission and monitoring.

Handles communication with the paddle_v3 OCR server.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class JobSubmitter:
    """HTTP client for interacting with paddle_v3 OCR server."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout: float = 300.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def submit_batch(
        self,
        pdf_paths: List[str],
        priority: int = 5,
        dpi: Optional[int] = None,
        preprocess: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """
        Submit a batch of PDFs for OCR processing.

        Args:
            pdf_paths: List of PDF file paths (container paths)
            priority: Job priority (higher = more important)
            dpi: Optional DPI override for rendering
            preprocess: Optional preprocessing config, e.g. {"clahe": {"clip": 2.5}}

        Returns:
            Job ID string

        Raises:
            Exception if submission fails
        """
        session = await self._get_session()

        payload = {
            "pdf_paths": pdf_paths,
            "priority": priority,
        }
        if dpi is not None:
            payload["dpi"] = dpi
        if preprocess is not None:
            payload["preprocess"] = preprocess

        async with session.post(
            f"{self.server_url}/process",
            json=payload,
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Submit failed ({response.status}): {text}")

            data = await response.json()
            job_id = data.get("job_id")

            if not job_id:
                raise Exception(f"No job_id in response: {data}")

            logger.debug(f"Submitted batch of {len(pdf_paths)} PDFs, job_id={job_id}")
            return job_id

    async def get_status(self, job_id: str) -> Dict:
        """
        Get job status.

        Args:
            job_id: Job ID to check

        Returns:
            Status dictionary with:
            - status: queued, processing, completed, failed, partial, completed_with_errors
            - total_pages: Total pages in job
            - processed_pages: Pages processed so far
            - failed_pages: Pages that failed OCR
            - start_time, end_time: Timestamps
        """
        session = await self._get_session()

        async with session.get(f"{self.server_url}/status/{job_id}") as response:
            if response.status == 404:
                raise Exception(f"Job not found: {job_id}")
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Status check failed ({response.status}): {text}")

            return await response.json()

    async def get_results(self, job_id: str) -> Dict:
        """
        Get job results.

        Args:
            job_id: Job ID to get results for

        Returns:
            Results dictionary with:
            - job_id: Job ID
            - status: Final status
            - total_pages: Total pages processed
            - process_time: Total processing time in seconds
            - results: Dict[pdf_path, Dict[page_num, page_result]]
        """
        session = await self._get_session()

        async with session.get(f"{self.server_url}/results/{job_id}") as response:
            if response.status == 404:
                raise Exception(f"Job not found: {job_id}")
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Results fetch failed ({response.status}): {text}")

            return await response.json()

    async def get_server_stats(self) -> Dict:
        """
        Get server statistics.

        Returns:
            Server stats including:
            - total_jobs: Jobs processed
            - completed_jobs: Completed jobs
            - total_pages_completed: Pages completed
            - avg_throughput_pages_per_sec: Average throughput
            - configuration: Server config
        """
        session = await self._get_session()

        async with session.get(f"{self.server_url}/stats") as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Stats fetch failed ({response.status}): {text}")

            return await response.json()

    async def health_check(self) -> bool:
        """
        Check if server is healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.server_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "healthy"
                return False
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> Dict:
        """
        Wait for a job to complete and return results.

        Args:
            job_id: Job ID to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait

        Returns:
            Final results dictionary

        Raises:
            TimeoutError if job doesn't complete in time
            Exception if job fails
        """
        start_time = asyncio.get_event_loop().time()
        completed_statuses = {"completed", "completed_with_errors", "partial", "failed"}

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

            status = await self.get_status(job_id)
            job_status = status.get("status", "unknown")

            if job_status in completed_statuses:
                if job_status == "failed":
                    raise Exception(f"Job {job_id} failed")
                return await self.get_results(job_id)

            await asyncio.sleep(poll_interval)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
