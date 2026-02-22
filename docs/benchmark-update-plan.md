# Plan: benchmark.py — comprehensive update for regression detection

## Context
A multiprocessing→threading refactor is imminent (`docs/threading-refactor-plan.md`). The benchmark needs to serve as the before/after regression detection tool. Currently it measures wall time and server time but misses warmup variability, per-page latencies, queue wait time, page-level errors, sample set identification, and has no way to compare against a baseline run. Memory peak calculation is also buggy.

## Changes

**File:** `benchmark.py`

### 1. Add warmup phase
- `--warmup` arg (default: 1 batch worth of docs).
- Before the timed run, submit warmup docs, wait for completion, discard results.
- Eliminates first-batch variability from cold GPU caches.

### 2. Save and identify the sample set
- `--sample-dir` arg (default: `benchmark_data/`). On first run, copy sampled PDFs into this directory. On subsequent runs, use the existing directory contents.
- Add `sample_files`, `seed`, `pdf_source_path` to saved JSON.

### 3. Capture queue wait time per job
- Extract `submit_time` and `start_time` from status dict.
- Compute `queue_wait = start_time - submit_time` per job.
- Report `avg_queue_wait` in summary.

### 4. Capture per-page latency stats
- From `/results/{job_id}`, extract each page's `process_time`.
- Compute p50, p95, p99.

### 5. Track page-level errors
- Count pages with `status: "error"` from results.
- Add `total_page_errors` to results.

### 6. Separate client overhead
- Per-job: `client_overhead = wall_time - server_process_time`.
- Report `avg_client_overhead`.

### 7. Batch-level throughput variance
- Compute pages/sec per batch.
- Report mean and std deviation.

### 8. GPU utilization sampling
- Background task polls `/health` every 2s, extracts GPU load.
- Reports average GPU utilization.

### 9. Measure startup time (`--measure-startup`)
- Restarts container, polls `/health` until healthy.
- Records `startup_time`.

### 10. Baseline comparison mode (`--compare`)
- Loads a previous run's JSON, prints comparison table with deltas.
- Exits with code 1 if throughput drops > threshold.

### 11. Expose render vs OCR timing breakdown
- Server change: add `render_time` and `ocr_time` to per-page result dict.
- Benchmark: extract and report avg render/OCR times.

### 12. Fix memory peak string comparison bug
- `parse_mem_usage()` helper for numeric comparison.

### 14. Minor: `if dpi:` → `if dpi is not None:`
