# paddle_v4_tensorrt

GPU-accelerated OCR server and production client for processing PDF documents using PP-OCRv4/v5 models via ONNX Runtime with TensorRT.

## Architecture

The system has two main components:

1. **Server** — A FastAPI service running inside Docker that renders PDF pages and performs OCR using direct ONNX Runtime inference with TensorRT acceleration. No RapidOCR or PaddlePaddle dependency.
2. **Client** — An async Python application that streams documents from a Supabase database, submits batches to the server, and writes results back.

### OCR Pipeline

```
PDF file
  → pypdfium2 renders page at configured DPI
  → Detection model locates text regions (DB postprocessing)
  → Text regions cropped and sorted top-to-bottom
  → Recognition model decodes text per crop (CTC greedy decode)
  → Results returned as structured JSON
```

The orientation classifier is **disabled** — it's unreliable at DPI=100 on thin crops and was the single biggest source of accuracy loss.

### Server Internals

- **Multiprocess workers**: N worker processes each run independent ONNX sessions on the GPU. Pages are distributed via a shared queue.
- **Shared memory buffer pool**: Pre-allocated fixed-size buffers (~18MB each) enable zero-copy page image transfer between processes.
- **TensorRT engine caching**: Detection and recognition models are compiled into optimized TRT engines at startup for a fixed set of input shapes (20 detection shapes, 25 recognition widths). Engines are cached in a Docker volume — first startup takes ~2 hours, subsequent restarts take seconds.
- **Width-quantized recognition**: Crop widths are rounded to the nearest 128px step (128–3200px range), limiting TRT to 25 engine variants instead of one per unique width.

## Quality

Validated against Google Cloud Vision API at 300 DPI as the gold standard:

| Dataset | TRT Accuracy | Paddle V3 Accuracy |
|---------|-------------|-------------------|
| 20-doc tuning set | 99.0% | 99.1% |
| 50-doc validation set | 98.9% (median 98.3%) | 99.1% (median 98.7%) |

The remaining ~1% gap is character-level recognition errors on degraded pre-2000 scans — a model limitation at DPI=100, not a pipeline issue.

See [FINDINGS.md](FINDINGS.md) for the full tuning journey and parameter experiments.

## Throughput

At DPI=100 with 10 workers on a single NVIDIA GPU:

- **13–16 pages/sec** sustained
- **88–98% GPU utilization** (GPU is the sole bottleneck)
- **~38% CPU utilization** (rendering is fast at ~0.02s/page)

| Workers | GPU Memory | Pages/sec |
|---------|-----------|-----------|
| 1 | 13.9 GB | 2.7 |
| 4 | 16.6 GB | ~10 |
| 8 | 20.9 GB | ~13 |
| 10 | 23.2 GB | 13–16 |

## Quick Start

### Start the server

```bash
cd paddle_v4_tensorrt
docker compose up -d
```

This builds the image (if needed) and starts the server on port **8003**. PDF files are mounted read-only from `/mnt/models/wake-county-pdfs`.

To use PP-OCRv4 models instead of v5:

```bash
docker compose -f docker-compose.v4.yml up -d
```

### Run the client

Process new (unprocessed) documents:

```bash
python3 -m paddle_v4_tensorrt.client.main --mode new
```

Reprocess existing documents:

```bash
python3 -m paddle_v4_tensorrt.client.main --mode reprocess
```

### Check server health

```bash
curl http://localhost:8003/health
```

## Server API

### `POST /process`

Submit PDFs for OCR processing.

```json
{
  "pdf_paths": ["/data/2024/01/15/DOC123_file.pdf"],
  "priority": 0,
  "dpi": 100
}
```

Returns `{ "job_id": "..." }`.

### `GET /status/{job_id}`

Poll job progress. Returns status (`queued`, `processing`, `completed`, `failed`, etc.), page counts, and timing.

### `GET /results/{job_id}`

Fetch completed results. Returns structured output keyed by PDF path and page number:

```json
{
  "results": {
    "/data/.../file.pdf": {
      "1": {
        "text_lines": [
          {"text": "...", "confidence": 0.95, "bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}
        ],
        "status": "success"
      }
    }
  }
}
```

### `GET /health`

Server health including GPU memory, temperature, CPU/memory usage, ORT version, and available providers.

### `GET /stats`

Global statistics: total jobs, completed jobs, total pages, average throughput.

## Client CLI

```
python3 -m paddle_v4_tensorrt.client.main [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--mode new\|reprocess` | Process unprocessed documents or reprocess existing ones |
| `--target N` | Stop after N documents |
| `--start-date YYYY-MM-DD` | Filter by recorded_at start date |
| `--end-date YYYY-MM-DD` | Filter by recorded_at end date |
| `--year YYYY` | Shorthand for a full year range |
| `--uploaded-after YYYY-MM-DD` | Filter by created_at start date |
| `--uploaded-before YYYY-MM-DD` | Filter by created_at end date |
| `--today` | Only documents uploaded today |
| `--document-type TYPE [TYPE ...]` | Filter by document type (e.g., `"DEED OF TRUST"`) |
| `--server URL` | OCR server URL (default: `http://localhost:8003`) |
| `--batch-size N` | Documents per batch (default: 16) |
| `--max-concurrent N` | Max concurrent batches (default: 4) |
| `--pdf-path PATH` | Host PDF directory |
| `--container-path PATH` | Container PDF mount point |
| `-v, --verbose` | Debug logging |
| `-q, --quiet` | Errors only |

### Client Features

- **Streaming pipeline**: Producer fetches documents via keyset pagination (500 per chunk), consumer assembles and submits batches. Bounded buffer provides automatic backpressure.
- **Zero-text fallback**: Documents that return no text at DPI=100 are automatically resubmitted at DPI=300.
- **Non-blocking DB writes**: Results are queued to a background worker so database saves don't stall job polling.
- **Circuit breaker**: Database pool opens the circuit after 5 consecutive failures and attempts recovery after 60 seconds.

## Configuration

### Server Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_WORKERS` | 4 | Worker processes (optimal: 8–10 for single GPU) |
| `DPI` | 100 | PDF rendering resolution |
| `BUFFER_POOL_SIZE` | 50 | Shared memory buffer count |
| `INSTANCE_ID` | `v4_tensorrt` | Server identifier |
| `REC_BATCH_SIZE` | 1 | Recognition batch size (1 is best for accuracy) |
| `DET_DB_THRESH` | 0.3 | Detection binarization threshold |
| `DET_DB_BOX_THRESH` | 0.5 | Minimum detection box confidence |
| `DET_DB_UNCLIP_RATIO` | 1.6 | Text region expansion factor |
| `DET_LIMIT_SIDE_LEN` | 1920 | Max detection input dimension |
| `TEXT_SCORE` | 0.3 | Minimum recognition confidence |
| `DET_MODEL_PATH` | (v4 det ONNX) | Detection model path |
| `REC_MODEL_PATH` | (v4 rec ONNX) | Recognition model path |
| `REC_KEYS_PATH` | (v1 keys) | Character dictionary path |
| `TRT_ENGINE_CACHE_PATH` | `/app/trt_cache` | TensorRT engine cache directory |

### Client Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_SERVER_URL` | `http://localhost:8003` | Server URL |
| `DATABASE_PASSWORD` | — | Supabase database password |
| `DB_MAX_CONNECTIONS` | 6 | Connection pool size |
| `HOST_PDF_PATH` | `/mnt/models/wake-county-pdfs` | PDF directory on host |
| `CONTAINER_PDF_PATH` | `/data` | PDF mount point inside container |
| `BATCH_SIZE` | 16 | Documents per submission batch |
| `MAX_CONCURRENT_BATCHES` | 4 | Parallel batch limit |
| `FALLBACK_DPI` | 300 | DPI for zero-text retry |

## Stack

- **Runtime**: NVIDIA CUDA 11.8, cuDNN 8, TensorRT 8.6
- **Inference**: ONNX Runtime 1.17.0 (TensorRT EP, FP16)
- **Models**: PP-OCRv5 mobile (det + rec) or PP-OCRv4 (det + rec + cls)
- **PDF rendering**: pypdfium2
- **Server**: FastAPI + uvicorn
- **Client**: asyncio + aiohttp + asyncpg
- **Database**: PostgreSQL (Supabase)

## Project Structure

```
paddle_v4_tensorrt/
├── server.py                  # FastAPI OCR server with multiprocess workers
├── Dockerfile                 # CUDA 11.8 + TensorRT 8.6 image
├── docker-compose.yml         # PP-OCRv5 models, port 8003
├── docker-compose.v4.yml      # PP-OCRv4 models, port 8004
├── FINDINGS.md                # Optimization results and tuning history
├── models/
│   ├── PP-OCRv4/              # v4 det, rec, cls ONNX models + keys
│   └── PP-OCRv5/              # v5 mobile det, rec ONNX models + keys
├── client/
│   ├── main.py                # CLI entry point
│   ├── client.py              # Main orchestrator (producer-consumer pipeline)
│   ├── config.py              # Configuration from env/CLI args
│   ├── database.py            # Async PostgreSQL pool + document operations
│   ├── job_submitter.py       # HTTP client for server API
│   ├── path_builder.py        # PDF path resolution and validation
│   └── result_processor.py    # OCR result extraction and formatting
├── compare_accuracy.py        # Benchmark: TRT vs Paddle V3
└── compare_mobile_vs_server.py # Benchmark: v5 mobile vs server models
```
