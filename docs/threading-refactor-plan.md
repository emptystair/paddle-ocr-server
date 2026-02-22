# Refactor: Multiprocessing Workers to Threaded Workers with Shared ONNX Sessions

## Context

The OCR server uses 8 multiprocessing workers, each loading its own copy of 3 ONNX models (~2.5GB each). This consumes ~21GB RAM at idle for duplicate model weights. Since the GPU is the actual bottleneck and ONNX Runtime sessions are thread-safe (GIL released during inference), we can share a single set of sessions across threads, reducing idle memory from ~21GB to ~3-4GB. This matters for cloud deployment cost.

Work is done in a git worktree (branch) so `main` stays untouched as the rollback path. Merge to `main` only after verification passes.

## Files to Modify

- `paddle_v4_tensorrt/server.py` — all server logic (single file, ~1170 lines)
- `paddle_v4_tensorrt/docker-compose.yml` — remove `shm_size`
- `paddle_v4_tensorrt/docker-compose.v4.yml` — remove `shm_size`
- `paddle_v4_tensorrt/README.md` — update architecture docs

## What Changes

### 1. Replace `PageOCRWorker` (mp.Process) with thread worker function

Delete the `PageOCRWorker` class. Replace with a `page_ocr_worker()` function that runs in a `threading.Thread`. Each thread gets a thread-local `PDFRenderer` (pypdfium2 is not thread-safe) but shares a single `OCREngine` instance passed in at creation.

### 2. Delete `SharedBufferPool`

No longer needed — threads share address space, so numpy arrays pass directly. The bounded `queue.Queue(maxsize=PAGE_QUEUE_SIZE)` provides backpressure instead.

### 3. Simplify `PDFRenderer`

Remove shared memory logic. `render_page()` returns `np.ndarray` directly instead of writing to a named shared memory buffer. Use `threading.local()` for per-thread renderer instances.

### 4. Simplify `PageTask` dataclass

Remove `buffer_name` and `img_shape` fields (shared memory artifacts).

### 5. Replace queues

| Current | New |
|---------|-----|
| `mp.Queue` (page_queue) | `queue.Queue(maxsize=PAGE_QUEUE_SIZE)` |
| `mp.Queue` (result_queue) | `queue.Queue()` |

`dispatch_queue`, `dispatcher_loop`, `process_results` — minimal changes (already thread-based, `queue.Empty` API is compatible).

### 6. Simplify `queue_pdf_pages()`

Remove buffer acquisition loop. Put `PageTask` directly on bounded queue (blocks when full = natural backpressure).

### 7. Update startup sequence

1. `warmup_trt_engines()` — unchanged
2. Create **one** shared `OCREngine` (not N)
3. Single cuDNN warmup pass (not N per-worker warmups)
4. Create `queue.Queue` instances
5. Start N daemon threads (no staggered 2s delays)
6. Start dispatcher + results threads — unchanged

### 8. Update shutdown

Signal `stop_event`, send N poison pills, join threads. No `.terminate()` needed.

### 9. Clean up imports/globals

Remove `multiprocessing`, `shared_memory` imports. Remove `buffer_manager` global, `BUFFER_SIZE` constant, `mp.set_start_method("spawn")`.

### 10. Docker compose updates

Remove `shm_size: 4gb` from both compose files. Remove `BUFFER_POOL_SIZE` env var.

## What Stays the Same

- All OCR pipeline functions (det_preprocess, det_postprocess, rec_preprocess, ctc_decode, etc.)
- `OCREngine.__init__` and `OCREngine.__call__`
- `create_ort_session`, `warmup_trt_engines`
- `JobManager` (already thread-safe)
- All FastAPI endpoints (same API contract)
- All client code (zero changes)
- TRT engine cache on disk (no recompilation)
- `WARMUP_DET_SHAPES`, `WARMUP_REC_WIDTHS`, shape snapping logic

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Idle RAM | ~21 GB | ~3-4 GB |
| Startup time | ~30s (staggered workers) | ~5s (single engine) |
| Throughput | 13-16 pages/sec | Same (GPU-bottlenecked) |
| GPU VRAM | ~12 GB | ~12 GB (unchanged) |

## Deployment

Before rebuilding the container, **stop the running OCR client** — it may be actively processing documents against the current server.

## Verification

1. Rebuild container: `docker compose build && docker compose up -d`
2. Check health: `curl localhost:8003/health` — should show healthy with N workers
3. Check memory: `docker stats paddle_v4_trt_server --no-stream` — should be ~3-4GB vs ~21GB
4. Submit test job and verify results contain text_lines
5. Run client benchmark (see `docs/benchmark.md`) and compare against pre-refactor baseline
6. Monitor throughput via `/stats` — should be comparable to before

## Risks

- **ONNX RT thread safety**: Documented as safe, but if issues arise we can add a `threading.Lock` around `.run()` calls as fallback
- **Error isolation**: Thread crash takes down the process (vs only one worker in multiprocessing). Mitigated by broad try/except in worker loop + Docker `restart: unless-stopped`
- **Throughput regression**: If GPU serialization via ORT's internal mutex is less efficient than process-level queuing, throughput could dip. Can test with 2 OCREngine instances as middle ground (still saves ~80% memory)
