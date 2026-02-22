# Cloud Deployment (Threaded Workers)

Memory-optimized variant of the OCR server using **threaded workers with shared ONNX sessions** instead of multiprocessing. Designed for cloud deployment where memory cost matters.

## Trade-offs vs Local (Multiprocessing)

| Metric | Local (multiprocessing) | Cloud (threaded) |
|--------|------------------------|------------------|
| Memory | ~21 GB | ~3 GB |
| Throughput | 13.4 pages/sec | 11.8 pages/sec |
| Memory efficiency | 0.6 pages/sec/GiB | 3.7 pages/sec/GiB |
| Startup time | ~30s | ~5s |
| GPU VRAM | ~12 GB | ~12 GB |

The ~12% throughput loss comes from shared CUDA context serialization (vs independent per-process contexts). The 6x memory efficiency improvement makes this the better choice for cloud instances priced by RAM.

## How It Works

N worker threads share `NUM_ENGINES` `OCREngine` instances (ONNX Runtime sessions are thread-safe — the GIL is released during GPU inference). Each thread gets its own `PDFRenderer` (pypdfium2 is not thread-safe). Workers are distributed round-robin across engines.

## Quick Start

```bash
cd paddle_v4_tensorrt/cloud
docker compose up -d
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_WORKERS` | 8 | Worker threads |
| `NUM_ENGINES` | 2 | Independent OCREngine instances (each with det+rec sessions) |
| `DPI` | 100 | PDF rendering resolution |
| `REC_BATCH_SIZE` | 1 | Recognition batch size |

Best tested config: 8 workers, 2 engines. More engines don't help (bottleneck is CUDA driver, not ORT mutex).

## API

Same API as the local server — see the main [README](../README.md#server-api).
