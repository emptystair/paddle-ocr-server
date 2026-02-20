"""
Database operations for OCR processing client.

Handles connection pooling, document fetching, and result storage.
Adapted from rapid_paddle/docker/ocr_db_pool.py
"""

import asyncio
import asyncpg
import logging
import time
import re
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"

    async def call(self, func, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
                logger.info("Circuit breaker entering half-open state")
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker is open (failures: {self.failure_count})"
                )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time
            and datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )

    def _on_success(self):
        self.failure_count = 0
        self.last_failure_time = None
        if self.state != "closed":
            self.state = "closed"
            logger.info("Circuit breaker closed - system recovered")

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(f"Circuit breaker opened - failures: {self.failure_count}")


class OCRDatabasePool:
    """Database connection pool for OCR processing."""

    def __init__(
        self,
        supabase_url: str,
        database_password: str,
        max_connections: int = 6,
    ):
        self.supabase_url = supabase_url
        self.database_password = database_password
        self.max_connections = max_connections
        self._pool: Optional[asyncpg.Pool] = None
        self.circuit_breaker = CircuitBreaker()
        self.stats = {
            "total_queries": 0,
            "query_errors": 0,
        }

    async def initialize(self):
        """Initialize the connection pool."""
        # Extract project ID from Supabase URL
        match = re.match(r'https://([^.]+)\.supabase\.co', self.supabase_url)
        if not match:
            raise ValueError(f"Invalid Supabase URL: {self.supabase_url}")

        project_id = match.group(1)
        logger.info(f"Connecting to Supabase project: {project_id}")

        self._pool = await asyncpg.create_pool(
            host="aws-0-us-east-2.pooler.supabase.com",
            port=5432,
            user=f"postgres.{project_id}",
            password=self.database_password,
            database="postgres",
            min_size=2,
            max_size=self.max_connections,
            max_queries=1000,
            max_inactive_connection_lifetime=300,
            command_timeout=300.0,
            statement_cache_size=0,  # Required for pgbouncer
            ssl="require",
        )
        logger.info(f"Database pool initialized (max {self.max_connections} connections)")

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Database pool closed")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self._pool:
            raise RuntimeError("Pool not initialized")

        async with self._pool.acquire() as conn:
            yield conn

    async def execute(self, query: str, *args) -> str:
        """Execute a query."""
        async with self.acquire() as conn:
            self.stats["total_queries"] += 1
            try:
                return await self.circuit_breaker.call(conn.execute, query, *args)
            except Exception:
                self.stats["query_errors"] += 1
                raise

    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Fetch rows from a query."""
        async with self.acquire() as conn:
            self.stats["total_queries"] += 1
            try:
                return await self.circuit_breaker.call(conn.fetch, query, *args)
            except Exception:
                self.stats["query_errors"] += 1
                raise

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single row."""
        async with self.acquire() as conn:
            self.stats["total_queries"] += 1
            try:
                return await self.circuit_breaker.call(conn.fetchrow, query, *args)
            except Exception:
                self.stats["query_errors"] += 1
                raise


class DocumentOperations:
    """High-level document operations for OCR processing."""

    def __init__(self, pool: OCRDatabasePool):
        self.pool = pool

    async def fetch_unprocessed_documents(
        self,
        limit: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        excluded_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Fetch documents WITHOUT ocr_results records.
        For "new" mode - processing fresh documents.
        """
        excluded = excluded_types or ["UNUSED FILE NUMBER", "BLANK"]

        params: List[Any] = []
        param_idx = 1

        # Excluded types as parameterized array
        params.append(excluded)
        excluded_param = f"${param_idx}::text[]"
        param_idx += 1

        date_filter = ""
        if start_date:
            date_filter += f" AND d.recorded_at >= ${param_idx}::timestamp"
            params.append(start_date)
            param_idx += 1

        if end_date:
            date_filter += f" AND d.recorded_at <= ${param_idx}::timestamp"
            params.append(end_date)
            param_idx += 1

        params.append(limit)

        query = f"""
            SELECT
                d.id,
                d.source_document_id,
                d.document_type,
                d.book_page_number,
                d.recorded_at,
                d.pdf_url,
                d.legal_description
            FROM documents d
            LEFT JOIN ocr_results o ON o.document_id = d.id
            WHERE o.id IS NULL
            AND d.pdf_url IS NOT NULL
            AND d.document_type != ALL({excluded_param})
            {date_filter}
            ORDER BY d.recorded_at DESC
            LIMIT ${param_idx}
        """

        rows = await self.pool.fetch(query, *params)
        return [dict(r) for r in rows]

    async def fetch_documents_for_reprocessing(
        self,
        limit: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        excluded_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Fetch documents that HAVE ocr_results but need reprocessing.
        For "reprocess" mode - improving existing OCR.
        """
        excluded = excluded_types or ["UNUSED FILE NUMBER", "BLANK"]

        params: List[Any] = []
        param_idx = 1

        # Excluded types as parameterized array
        params.append(excluded)
        excluded_param = f"${param_idx}::text[]"
        param_idx += 1

        date_filter = ""
        if start_date:
            date_filter += f" AND d.recorded_at >= ${param_idx}::timestamp"
            params.append(start_date)
            param_idx += 1

        if end_date:
            date_filter += f" AND d.recorded_at <= ${param_idx}::timestamp"
            params.append(end_date)
            param_idx += 1

        params.append(limit)

        query = f"""
            SELECT
                d.id,
                d.source_document_id,
                d.document_type,
                d.book_page_number,
                d.recorded_at,
                d.pdf_url,
                d.legal_description,
                o.id as ocr_result_id
            FROM documents d
            INNER JOIN ocr_results o ON o.document_id = d.id
            WHERE d.pdf_url IS NOT NULL
            AND d.document_type != ALL({excluded_param})
            {date_filter}
            ORDER BY d.recorded_at DESC
            LIMIT ${param_idx}
        """

        rows = await self.pool.fetch(query, *params)
        return [dict(r) for r in rows]

    async def save_ocr_results(self, results: List[Dict]) -> int:
        """
        Save OCR results to database using bulk COPY + INSERT ON CONFLICT.

        Uses a temp table and COPY protocol for much faster batch inserts
        compared to individual INSERT statements.

        Args:
            results: List of dicts with document_id, full_text, pages, confidence, processing_time

        Returns:
            Number of rows saved
        """
        if not results:
            return 0

        import json as json_mod
        import uuid as uuid_mod

        # Prepare records
        records = []
        for r in results:
            doc_id = r['document_id']
            if isinstance(doc_id, str):
                doc_id = uuid_mod.UUID(doc_id)

            # Ensure ocr_metadata is a JSON string
            metadata = r.get('ocr_metadata', '{}')
            if isinstance(metadata, dict):
                metadata = json_mod.dumps(metadata)

            records.append((
                doc_id,
                r.get('full_text', ''),
                r.get('pages', 0),
                float(r.get('confidence', 95.0)),
                float(r.get('processing_time', 0)),
                metadata,
            ))

        temp_table = f"ocr_temp_{int(time.time() * 1000)}"

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f"""
                    CREATE TEMP TABLE {temp_table} (
                        document_id UUID,
                        full_text TEXT,
                        pages INTEGER,
                        confidence NUMERIC,
                        processing_time NUMERIC,
                        ocr_metadata TEXT
                    ) ON COMMIT DROP
                """)

                await conn.copy_records_to_table(
                    temp_table,
                    records=records,
                    columns=['document_id', 'full_text', 'pages',
                             'confidence', 'processing_time', 'ocr_metadata'],
                )

                await conn.execute(f"""
                    INSERT INTO ocr_results (
                        document_id, full_text, pages, confidence,
                        processing_time, ocr_metadata, created_at, updated_at
                    )
                    SELECT document_id, full_text, pages, confidence,
                           processing_time, ocr_metadata::jsonb, NOW(), NOW()
                    FROM {temp_table}
                    ON CONFLICT (document_id) DO UPDATE SET
                        full_text = EXCLUDED.full_text,
                        pages = EXCLUDED.pages,
                        confidence = EXCLUDED.confidence,
                        processing_time = EXCLUDED.processing_time,
                        ocr_metadata = COALESCE(ocr_results.ocr_metadata, '{{}}'::jsonb) || EXCLUDED.ocr_metadata,
                        updated_at = NOW()
                """)

        return len(records)

    async def get_document_count(
        self,
        mode: str = "new",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        excluded_types: Optional[List[str]] = None,
        included_types: Optional[List[str]] = None,
        uploaded_after: Optional[str] = None,
        uploaded_before: Optional[str] = None,
    ) -> int:
        """Get count of documents matching criteria.

        Uses fast approximate count (pg_class stats) when no filters are
        applied in 'new' mode, since the exact COUNT query scans millions
        of rows. Falls back to exact count when filters are present.
        """
        has_filters = any([start_date, end_date, included_types,
                          uploaded_after, uploaded_before])

        # Fast path: approximate count from pg_class stats (~0.1s vs ~25s)
        if mode == "new" and not has_filters:
            row = await self.pool.fetchrow("""
                SELECT
                    (SELECT reltuples::bigint FROM pg_class WHERE relname = 'documents') -
                    (SELECT reltuples::bigint FROM pg_class WHERE relname = 'ocr_results')
                    AS approx_missing
            """)
            return max(row["approx_missing"], 0) if row else 0

        # Exact count when filters are applied
        excluded = excluded_types or ["UNUSED FILE NUMBER", "BLANK"]
        f = QueryFilter()

        f.add("d.document_type != ALL(${idx}::text[])", excluded)

        if included_types:
            f.add("d.document_type = ANY(${idx}::text[])", included_types)
        if start_date:
            f.add("d.recorded_at >= ${idx}::timestamp", start_date)
        if end_date:
            f.add("d.recorded_at <= ${idx}::timestamp", end_date)
        if uploaded_after:
            f.add("d.created_at >= ${idx}::timestamp", uploaded_after)
        if uploaded_before:
            f.add("d.created_at < (${idx}::timestamp + interval '1 day')", uploaded_before)

        if mode == "new":
            join_clause = ""
            f.add_raw("NOT EXISTS (SELECT 1 FROM ocr_results o WHERE o.document_id = d.id)")
        else:
            join_clause = "INNER JOIN ocr_results o ON o.document_id = d.id"

        query = f"""
            SELECT COUNT(*) FROM documents d
            {join_clause}
            WHERE d.pdf_url IS NOT NULL
            {f.where_clause}
        """

        row = await self.pool.fetchrow(query, *f.params)
        return row[0] if row else 0

    async def fetch_documents_chunked(
        self,
        mode: str = "new",
        chunk_size: int = 500,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        excluded_types: Optional[List[str]] = None,
        included_types: Optional[List[str]] = None,
        uploaded_after: Optional[str] = None,
        uploaded_before: Optional[str] = None,
        cursor_recorded_at: Optional[Any] = None,
        cursor_id: Optional[Any] = None,
    ) -> Tuple[List[Dict], Optional[Any], Optional[Any]]:
        """
        Fetch a chunk of documents using keyset pagination.

        Uses (recorded_at, id) as the cursor for stable ordering
        even when multiple documents share the same recorded_at.

        Args:
            mode: "new" for unprocessed, "reprocess" for existing
            chunk_size: Number of documents per chunk
            start_date: Filter recorded_at >= date
            end_date: Filter recorded_at <= date
            excluded_types: Document types to exclude
            included_types: Document types to include (overrides exclusion)
            uploaded_after: Filter created_at >= date
            uploaded_before: Filter created_at <= date
            cursor_recorded_at: Keyset cursor - recorded_at from last row
            cursor_id: Keyset cursor - id from last row

        Returns:
            (documents, next_cursor_recorded_at, next_cursor_id)
            Cursors are None when no more documents available.
        """
        excluded = excluded_types or ["UNUSED FILE NUMBER", "BLANK"]
        f = QueryFilter()

        f.add("d.document_type != ALL(${idx}::text[])", excluded)

        if included_types:
            f.add("d.document_type = ANY(${idx}::text[])", included_types)
        if start_date:
            f.add("d.recorded_at >= ${idx}::timestamp", start_date)
        if end_date:
            f.add("d.recorded_at <= ${idx}::timestamp", end_date)
        if uploaded_after:
            f.add("d.created_at >= ${idx}::timestamp", uploaded_after)
        if uploaded_before:
            f.add("d.created_at < (${idx}::timestamp + interval '1 day')", uploaded_before)

        # Sort order: DESC for "new" mode (newest unprocessed docs are
        # most important for the database); ASC for "reprocess" (oldest
        # existing OCR results benefit most from reprocessing)
        if mode == "new":
            sort_order = "DESC"
            cursor_op = "<"
        else:
            sort_order = "ASC"
            cursor_op = ">"

        if cursor_recorded_at is not None and cursor_id is not None:
            f.add_double(
                f"(d.recorded_at, d.id) {cursor_op} (${{idx1}}::timestamptz, ${{idx2}}::uuid)",
                cursor_recorded_at, cursor_id
            )

        # Build join based on mode
        if mode == "new":
            join_clause = ""
            f.add_raw("NOT EXISTS (SELECT 1 FROM ocr_results o WHERE o.document_id = d.id)")
            extra_cols = ""
        else:
            join_clause = "INNER JOIN ocr_results o ON o.document_id = d.id"
            extra_cols = ", o.id as ocr_result_id"

        limit_idx = f.next_idx
        all_params = f.params + [chunk_size]

        query = f"""
            SELECT
                d.id,
                d.source_document_id,
                d.document_type,
                d.book_page_number,
                d.recorded_at,
                d.pdf_url,
                d.legal_description
                {extra_cols}
            FROM documents d
            {join_clause}
            WHERE d.pdf_url IS NOT NULL
            {f.where_clause}
            ORDER BY d.recorded_at {sort_order}, d.id {sort_order}
            LIMIT ${limit_idx}
        """

        rows = await self.pool.fetch(query, *all_params)
        documents = [dict(r) for r in rows]

        # Determine next cursor
        if len(documents) < chunk_size:
            return documents, None, None

        last = documents[-1]
        return documents, last['recorded_at'], last['id']


class QueryFilter:
    """Builds parameterized WHERE clauses incrementally.

    Tracks parameter indices for asyncpg $N placeholders and
    accumulates conditions and parameter values.
    """

    def __init__(self, param_start: int = 1):
        self._conditions: List[str] = []
        self._params: List[Any] = []
        self._idx = param_start

    def add(self, sql_template: str, value: Any):
        """Add a condition with one parameter. Template uses {idx} placeholder."""
        self._conditions.append(sql_template.replace("{idx}", str(self._idx)))
        # asyncpg requires actual date/datetime objects, not strings
        if "::timestamp" in sql_template and isinstance(value, str):
            value = date.fromisoformat(value)
        self._params.append(value)
        self._idx += 1

    def add_double(self, sql_template: str, value1: Any, value2: Any):
        """Add a condition with two parameters. Template uses {idx1} and {idx2}."""
        rendered = sql_template.replace("{idx1}", str(self._idx)).replace("{idx2}", str(self._idx + 1))
        self._conditions.append(rendered)
        self._params.extend([value1, value2])
        self._idx += 2

    def add_raw(self, condition: str):
        """Add a condition with no parameters."""
        self._conditions.append(condition)

    @property
    def where_clause(self) -> str:
        """Return conditions as 'AND cond1 AND cond2 ...' string."""
        if not self._conditions:
            return ""
        return "AND " + " AND ".join(self._conditions)

    @property
    def params(self) -> list:
        return list(self._params)

    @property
    def next_idx(self) -> int:
        return self._idx
