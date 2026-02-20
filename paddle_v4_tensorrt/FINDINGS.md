# PP-OCRv5 + TensorRT Pipeline: Optimization Findings

## Overview

This pipeline runs PP-OCRv5 models (det + rec) through ONNX Runtime's TensorRT Execution Provider for GPU-accelerated OCR. The classifier (orientation detection) is disabled. Quality is validated against Google Cloud Vision API at 300 DPI as the gold standard.

## Architecture

- **Framework**: ONNX Runtime 1.17.0 with TensorRT EP (FP16)
- **Models**: PP-OCRv5 mobile det (ONNX) + PP-OCRv5 mobile rec (ONNX)
- **Pipeline**: PDF render (pypdfium2) → detection (CUDA EP) → recognition (TRT EP) → CTC decode
- **Parallelism**: N worker processes, each with independent ONNX sessions, sharing a page queue
- **TRT Engine Caching**: Compiled engines persist in a Docker volume (`trt_cache`), eliminating recompilation on restart

## Quality Results

### Gold Standard Comparison (Google Cloud Vision at 300 DPI)

**20-doc tuning set (seed=42):**
- TRT: 99.0% of gold standard character count
- Paddle V3 (native PaddlePaddle): 99.1% of gold standard

**50-doc validation set (seed=99):**
- TRT: 98.9% of gold standard (median 98.3%, range 80.8%–100.7%)
- Paddle V3: 99.1% of gold standard (median 98.7%, range 82.3%–100.6%)
- Docs below 90%: TRT=1, V3=2

### Quality Tuning Journey

| Change | TRT vs Gold | Notes |
|--------|-------------|-------|
| Initial (default params) | ~89% | Default TEXT_SCORE=0.5, unclip=2.0, adaptive limit |
| TEXT_SCORE 0.5 → 0.3 | 94.8% | Recovered low-confidence but correct text |
| DET_DB_UNCLIP_RATIO 2.0 → 1.6 | 97.5% | Tighter text regions, matches V3 setting |
| DET_LIMIT_SIDE_LEN adaptive → 1920 | 97.5% | Fixed limit matching V3 |
| DET_DB_BOX_THRESH 0.5 → 0.4 | No change | Detection thresholds not the bottleneck at DPI=100 |
| DPI 100 → 150 | 95.1% (worse) | Larger images change detection behavior |
| CLS_THRESH 0.5 (enable classifier) | 89.1% (worse) | Classifier confidently wrong on thin crops at low DPI |
| **Disable classifier entirely** | **99.0%** | Matches V3's `use_textline_orientation=False` |

### Key Quality Insight

The orientation classifier is unreliable at DPI=100. It confidently predicts `pred_idx=0` (no rotation needed) even for upside-down text, actively flipping correctly-oriented crops. Paddle V3 disables it entirely (`use_textline_orientation=False`), and doing the same here was the single biggest quality improvement.

### Nature of Remaining Misses

The ~1% gap vs gold is primarily **character-level recognition errors** on degraded documents:
- Old scanned documents (pre-2000) with poor scan quality
- Form fill-in fields with handwriting mixed into printed text
- Both TRT and V3 have the same issues on these documents — it's a model limitation at DPI=100, not a pipeline issue

## Throughput Results

### Worker Scaling (DPI=100, REC_BATCH_SIZE=1)

| Workers | GPU Mem (GB) | Pages/sec | Efficiency |
|---------|-------------|-----------|------------|
| 1 | 13.9 | 2.7 | 100% |
| 2 | 14.2 | ~5 | 93% |
| 4 | 16.6 | ~10 | 93% |
| 6 | 18.7 | ~12 | 74% |
| 8 | 20.9 | ~13 | 60% |
| 10 | 23.2 | 13.4–15.8 | 50–59% |

### Profiling (10 workers)

- **GPU utilization**: 88–98% (saturated)
- **CPU utilization**: ~38% (not the bottleneck)
- **Per-page breakdown**: render=0.02s, OCR=0.2–1.7s (GPU-bound)
- **Effective GPU efficiency**: ~95% — workers queue for GPU inference time

### Batching Experiments

| Config | Workers | Batch | Width Step | Pages/sec | Accuracy |
|--------|---------|-------|------------|-----------|----------|
| Baseline | 10 | 1 | 128 | 13.4–15.8 | 98.9% |
| Naive batch=6 | 10 | 6 | 128 | 44.9 | 9.9% (broken) |
| Width-grouped batch=6 | 10 | 6 | 128 | OOM | — |
| Width-grouped batch=6 | 6 | 6 | 128 | 12.2 | 99.0% |
| Width-grouped batch=4 | 6 | 4 | 320 | — | 98.2% (accuracy drop) |

**Why batching doesn't help:**
1. **Naive batching destroys accuracy** — crops have different widths; padding to the widest in the batch causes the rec model to produce garbage on excessively padded inputs
2. **Width-grouped batching preserves accuracy** but groups are too small — ~46 crops/page across ~25 width buckets = ~1–2 crops per group, so batch_size>1 rarely activates
3. **Coarser quantization (320-step)** creates larger groups but hurts accuracy by 1.5% due to excessive padding
4. **Batch>1 requires more GPU memory** — can't fit 10 workers with batched inference, must reduce to 6 workers, losing more throughput than batching gains

### Throughput Bottleneck Analysis

The GPU is the sole bottleneck. Each of the 10 worker processes creates independent ONNX Runtime sessions (det + rec), and all contend for GPU inference time. The pipeline is already at ~95% GPU efficiency.

**Options for further throughput gains (not yet implemented):**
- **Multi-GPU**: Add a second GPU and split workers across GPUs
- **Centralized inference queue**: Workers handle CPU-bound rendering and crop extraction, then submit crops to a single inference process that batches efficiently — avoids 10 separate GPU contexts competing
- **Adaptive DPI**: Run all docs at DPI=100 first, then re-run low-confidence docs at higher DPI — improves quality without hurting throughput on the 99% of docs that work fine at DPI=100

## Current Optimal Configuration

```yaml
NUM_WORKERS: 10
DPI: 100
REC_BATCH_SIZE: 1
REC_WIDTH_STEP: 128     # width quantization for TRT engine caching
DET_DB_THRESH: 0.3
DET_DB_BOX_THRESH: 0.5
DET_DB_UNCLIP_RATIO: 1.6
DET_LIMIT_SIDE_LEN: 1920
TEXT_SCORE: 0.3
Classifier: DISABLED
```

## Detection Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| DET_DB_THRESH | 0.3 | Binarization threshold for probability map |
| DET_DB_BOX_THRESH | 0.5 | Minimum confidence to keep a detected box |
| DET_DB_UNCLIP_RATIO | 1.6 | Text region expansion factor (lower = tighter boxes) |
| DET_LIMIT_SIDE_LEN | 1920 | Max image dimension for detection (fixed, not adaptive) |
| TEXT_SCORE | 0.3 | Minimum recognition confidence to include in output |

## Recognition Pipeline Details

- **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Input height**: 48px (fixed)
- **Width quantization**: 128px steps, max 3200px — ensures TRT engine reuse
- **Decoding**: CTC greedy decode with blank=0
- **Max rec width**: 3200px (crops wider than this are capped)

## TRT Engine Warmup

First startup compiles TRT engines for all 25 quantized widths (128, 256, ..., 3200). This takes ~2 hours but only happens once — engines are cached in a Docker volume. Subsequent startups load cached engines in seconds.
