#!/usr/bin/env python3
"""
OCR Server Benchmark — measures throughput, memory, and latency.

Bypasses the database entirely. Discovers PDFs on disk, submits them
directly to the server API, and reports performance metrics.

Usage:
    python benchmark.py                     # 100 docs, default settings
    python benchmark.py --docs 500          # 500 docs
    python benchmark.py --batch-size 32     # larger batches
    python benchmark.py --save results.json # save results to file
    python benchmark.py --compare old.json  # compare against baseline
    python benchmark.py --measure-startup   # include container startup time
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import aiohttp

# Paths
HOST_PDF_PATH = "/mnt/models/wake-county-pdfs"
CONTAINER_PDF_PATH = "/data"
SERVER_URL = "http://localhost:8003"
DOCKER_CONTAINER = "paddle_v4_trt_server"

# Polling
POLL_INTERVAL = 0.5
JOB_TIMEOUT = 600

# GPU sampling
GPU_SAMPLE_INTERVAL = 2.0


def discover_pdfs(root: str, count: int, seed: int = 42) -> List[str]:
    """Find PDFs on disk and return a random sample."""
    pdfs = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".pdf"):
                pdfs.append(os.path.join(dirpath, f))
    if not pdfs:
        print(f"No PDFs found in {root}")
        sys.exit(1)
    random.seed(seed)
    sample = random.sample(pdfs, min(count, len(pdfs)))
    print(f"Discovered {len(pdfs):,} PDFs, sampled {len(sample)}")
    return sample


def load_sample_manifest(sample_dir: str) -> Optional[List[str]]:
    """Load host paths from a saved manifest file."""
    manifest = os.path.join(sample_dir, "manifest.json")
    if not os.path.isfile(manifest):
        return None
    with open(manifest) as f:
        data = json.load(f)
    paths = data.get("host_paths", [])
    # Verify at least one path still exists
    if paths and os.path.isfile(paths[0]):
        return paths
    print(f"WARNING: Manifest paths no longer valid (e.g. {paths[0]})")
    return None


def save_sample_manifest(host_paths: List[str], sample_dir: str,
                         seed: int, source_path: str):
    """Save a manifest recording which host paths were sampled."""
    os.makedirs(sample_dir, exist_ok=True)
    manifest = os.path.join(sample_dir, "manifest.json")
    with open(manifest, "w") as f:
        json.dump({
            "seed": seed,
            "source_path": source_path,
            "count": len(host_paths),
            "host_paths": host_paths,
        }, f, indent=2)


def to_container_path(host_path: str) -> str:
    """Convert host PDF path to container mount path."""
    rel = os.path.relpath(host_path, HOST_PDF_PATH)
    return os.path.join(CONTAINER_PDF_PATH, rel)


def get_docker_memory() -> Optional[str]:
    """Get current memory usage from docker stats."""
    try:
        result = subprocess.run(
            ["docker", "stats", DOCKER_CONTAINER, "--no-stream",
             "--format", "{{.MemUsage}}"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return None


def parse_mem_usage(s: str) -> float:
    """Parse docker memory string like '3.8GiB / 62.5GiB' to GiB float."""
    if not s:
        return 0.0
    # Extract the usage part (before ' / ')
    usage_part = s.split("/")[0].strip()
    match = re.match(r"([\d.]+)\s*(GiB|MiB|KiB|B)", usage_part, re.IGNORECASE)
    if not match:
        return 0.0
    val = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "mib":
        val /= 1024
    elif unit == "kib":
        val /= 1024 * 1024
    elif unit == "b":
        val /= 1024 * 1024 * 1024
    return val


@dataclass
class JobResult:
    job_id: str
    num_pdfs: int
    total_pages: int
    submit_time: float
    complete_time: float
    wall_time: float
    server_process_time: float
    status: str
    errors: int = 0
    queue_wait: float = 0.0
    client_overhead: float = 0.0
    page_times: List[float] = field(default_factory=list)
    page_render_times: List[float] = field(default_factory=list)
    page_ocr_times: List[float] = field(default_factory=list)
    page_errors: int = 0


@dataclass
class BenchmarkResults:
    timestamp: str = ""
    server_url: str = ""
    num_documents: int = 0
    batch_size: int = 0
    max_concurrent: int = 0
    total_pages: int = 0
    total_wall_time: float = 0.0
    total_server_time: float = 0.0
    pages_per_sec_wall: float = 0.0
    pages_per_sec_server: float = 0.0
    docs_per_sec: float = 0.0
    jobs_submitted: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    memory_before: str = ""
    memory_after: str = ""
    memory_peak: str = ""
    job_results: List[dict] = field(default_factory=list)
    # New fields
    seed: int = 42
    pdf_source_path: str = ""
    sample_dir: str = ""
    sample_files: List[str] = field(default_factory=list)
    warmup_docs: int = 0
    warmup_time: float = 0.0
    avg_queue_wait: float = 0.0
    avg_client_overhead: float = 0.0
    page_latency_p50: float = 0.0
    page_latency_p95: float = 0.0
    page_latency_p99: float = 0.0
    total_page_errors: int = 0
    batch_throughput_mean: float = 0.0
    batch_throughput_std: float = 0.0
    avg_gpu_utilization: float = 0.0
    gpu_samples: int = 0
    startup_time: float = 0.0
    avg_render_time: float = 0.0
    avg_ocr_time: float = 0.0


def percentile(sorted_data: List[float], p: float) -> float:
    """Compute the p-th percentile of sorted data (0-100)."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


async def submit_job(session: aiohttp.ClientSession, server_url: str,
                     pdf_paths: List[str], dpi: Optional[int] = None) -> str:
    """Submit a batch of PDFs and return job_id."""
    payload = {"pdf_paths": pdf_paths, "priority": 0}
    if dpi is not None:
        payload["dpi"] = dpi
    async with session.post(f"{server_url}/process", json=payload) as resp:
        data = await resp.json()
        return data["job_id"]


async def wait_for_job(session: aiohttp.ClientSession, server_url: str,
                       job_id: str, timeout: float = JOB_TIMEOUT) -> dict:
    """Poll until job completes or times out."""
    terminal = {"completed", "completed_with_errors", "partial", "failed"}
    t0 = time.time()
    while time.time() - t0 < timeout:
        async with session.get(f"{server_url}/status/{job_id}") as resp:
            data = await resp.json()
            if data.get("status") in terminal:
                return data
        await asyncio.sleep(POLL_INTERVAL)
    return {"status": "timeout", "job_id": job_id}


async def get_results(session: aiohttp.ClientSession, server_url: str,
                      job_id: str) -> dict:
    """Fetch completed job results."""
    async with session.get(f"{server_url}/results/{job_id}") as resp:
        return await resp.json()


async def measure_startup(server_url: str) -> float:
    """Restart container and measure time to healthy."""
    print(f"\nMeasuring startup time (restarting {DOCKER_CONTAINER})...")
    subprocess.run(["docker", "restart", DOCKER_CONTAINER],
                   capture_output=True, timeout=30)
    t0 = time.time()
    while time.time() - t0 < 120:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{server_url}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") == "healthy":
                            elapsed = time.time() - t0
                            print(f"Startup time: {elapsed:.1f}s")
                            return elapsed
        except Exception:
            pass
        await asyncio.sleep(0.5)
    print("WARNING: Server did not become healthy within 120s")
    return -1.0


async def gpu_sampler(session: aiohttp.ClientSession, server_url: str,
                      samples: List[float], stop_event: asyncio.Event):
    """Background task to sample GPU utilization from /health."""
    while not stop_event.is_set():
        try:
            async with session.get(f"{server_url}/health",
                                   timeout=aiohttp.ClientTimeout(total=5)) as resp:
                data = await resp.json()
                gpu_stats = data.get("gpu_stats", [])
                if gpu_stats:
                    load_str = gpu_stats[0].get("load", "0%")
                    load_val = float(load_str.rstrip("%"))
                    samples.append(load_val)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=GPU_SAMPLE_INTERVAL)
        except asyncio.TimeoutError:
            pass


async def run_benchmark(args):
    """Main benchmark loop."""
    server_url = args.server
    batch_size = args.batch_size
    max_concurrent = args.max_concurrent

    # Measure startup if requested (before anything else)
    startup_time = 0.0
    if args.measure_startup:
        startup_time = await measure_startup(server_url)

    # Determine PDF source: fixed manifest or discover from source
    sample_dir = args.sample_dir
    existing_paths = load_sample_manifest(sample_dir)

    if existing_paths:
        print(f"Using fixed sample set from {sample_dir}/ ({len(existing_paths)} PDFs)")
        host_paths = existing_paths
    else:
        host_paths = discover_pdfs(HOST_PDF_PATH, args.docs, seed=args.seed)
        save_sample_manifest(host_paths, sample_dir, args.seed, HOST_PDF_PATH)
        print(f"Saved sample manifest to {sample_dir}/ for reproducibility")

    sample_filenames = [os.path.basename(p) for p in host_paths]
    container_paths = [to_container_path(p) for p in host_paths]

    # Split into batches
    batches = []
    for i in range(0, len(container_paths), batch_size):
        batches.append(container_paths[i:i + batch_size])
    print(f"Split into {len(batches)} batches of up to {batch_size} docs")

    # Check server health
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{server_url}/health") as resp:
                health = await resp.json()
                print(f"Server: {health.get('engine', 'unknown')}")
                print(f"Workers: {health.get('workers', '?')}")
                print(f"GPU: {health.get('gpu_stats', [{}])[0].get('name', '?')}")
        except Exception as e:
            print(f"Cannot reach server at {server_url}: {e}")
            sys.exit(1)

    # Warmup phase
    warmup_batches = args.warmup
    warmup_time = 0.0
    if warmup_batches > 0 and len(batches) > 0:
        warmup_docs_count = min(warmup_batches * batch_size, len(container_paths))
        warmup_paths = container_paths[:warmup_docs_count]
        warmup_batch_list = []
        for i in range(0, len(warmup_paths), batch_size):
            warmup_batch_list.append(warmup_paths[i:i + batch_size])

        print(f"Warmup: {len(warmup_paths)} docs...", end="", flush=True)
        t_warmup_start = time.time()
        async with aiohttp.ClientSession() as session:
            for wb in warmup_batch_list:
                job_id = await submit_job(session, server_url, wb)
                await wait_for_job(session, server_url, job_id)
        warmup_time = time.time() - t_warmup_start
        print(f"done ({warmup_time:.1f}s)")

    # Capture baseline memory
    mem_before = get_docker_memory()
    print(f"Memory before: {mem_before}")

    # Reset server stats baseline
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{server_url}/stats") as resp:
            stats_before = await resp.json()

    # Prepare results
    results = BenchmarkResults(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        server_url=server_url,
        num_documents=len(host_paths),
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        seed=args.seed,
        pdf_source_path=HOST_PDF_PATH,
        sample_dir=sample_dir,
        sample_files=sample_filenames,
        warmup_docs=warmup_batches * batch_size if warmup_batches > 0 else 0,
        warmup_time=warmup_time,
        startup_time=startup_time,
    )

    job_results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    mem_samples = [(mem_before, parse_mem_usage(mem_before or ""))] if mem_before else []
    batch_throughputs = []

    # Start GPU sampler
    gpu_samples: List[float] = []
    gpu_stop = asyncio.Event()

    async with aiohttp.ClientSession() as gpu_session:
        gpu_task = asyncio.create_task(
            gpu_sampler(gpu_session, server_url, gpu_samples, gpu_stop)
        )

        async def process_batch(batch: List[str], batch_idx: int):
            async with semaphore:
                async with aiohttp.ClientSession() as session:
                    t_submit = time.time()
                    job_id = await submit_job(session, server_url, batch)

                    status = await wait_for_job(session, server_url, job_id)
                    t_complete = time.time()

                    # Sample memory during processing
                    mem = get_docker_memory()
                    if mem:
                        mem_samples.append((mem, parse_mem_usage(mem)))

                    total_pages = status.get("total_pages", 0)
                    failed = status.get("failed_pages", 0)

                    # Queue wait time
                    queue_wait = 0.0
                    s_submit = status.get("submit_time")
                    s_start = status.get("start_time")
                    if s_submit is not None and s_start is not None:
                        try:
                            queue_wait = float(s_start) - float(s_submit)
                        except (TypeError, ValueError):
                            pass

                    # Get detailed results
                    server_time = 0.0
                    page_times = []
                    page_render_times = []
                    page_ocr_times = []
                    page_errors = 0
                    if status.get("status") in ("completed", "completed_with_errors", "partial"):
                        try:
                            res = await get_results(session, server_url, job_id)
                            server_time = res.get("process_time", 0.0)

                            # Extract per-page data
                            # results is {pdf_path: {page_num: page_result}}
                            results_dict = res.get("results", {})
                            for pdf_path, pages in results_dict.items():
                                if not isinstance(pages, dict):
                                    continue
                                for page_num, page_result in pages.items():
                                    if not isinstance(page_result, dict):
                                        continue
                                    pt = page_result.get("process_time")
                                    if pt is not None:
                                        page_times.append(float(pt))
                                    rt = page_result.get("render_time")
                                    if rt is not None:
                                        page_render_times.append(float(rt))
                                    ot = page_result.get("ocr_time")
                                    if ot is not None:
                                        page_ocr_times.append(float(ot))
                                    if page_result.get("status") == "error":
                                        page_errors += 1
                        except Exception:
                            pass

                    wall_time = t_complete - t_submit
                    client_overhead = wall_time - server_time if server_time > 0 else 0.0

                    # Batch throughput
                    if total_pages > 0 and wall_time > 0:
                        batch_throughputs.append(total_pages / wall_time)

                    jr = JobResult(
                        job_id=job_id,
                        num_pdfs=len(batch),
                        total_pages=total_pages,
                        submit_time=t_submit,
                        complete_time=t_complete,
                        wall_time=wall_time,
                        server_process_time=server_time,
                        status=status.get("status", "unknown"),
                        errors=failed,
                        queue_wait=queue_wait,
                        client_overhead=client_overhead,
                        page_times=page_times,
                        page_render_times=page_render_times,
                        page_ocr_times=page_ocr_times,
                        page_errors=page_errors,
                    )
                    job_results.append(jr)

                    status_str = status.get("status", "?")
                    print(f"  Batch {batch_idx + 1}/{len(batches)}: {total_pages} pages, "
                          f"{wall_time:.1f}s wall, {server_time:.1f}s server, "
                          f"status={status_str}")

        print(f"\nRunning benchmark ({max_concurrent} concurrent batches)...")
        t_start = time.time()

        tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
        await asyncio.gather(*tasks)

        t_end = time.time()

        # Stop GPU sampler
        gpu_stop.set()
        await gpu_task

    total_wall = t_end - t_start

    # Capture final memory
    mem_after = get_docker_memory()
    print(f"Memory after: {mem_after}")

    # Get server stats delta
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{server_url}/stats") as resp:
            stats_after = await resp.json()

    # Compute results
    total_pages = sum(jr.total_pages for jr in job_results)
    total_server_time = sum(jr.server_process_time for jr in job_results)
    completed = sum(1 for jr in job_results if jr.status in ("completed", "completed_with_errors"))
    failed = sum(1 for jr in job_results if jr.status in ("failed", "timeout"))

    results.total_pages = total_pages
    results.total_wall_time = total_wall
    results.total_server_time = total_server_time
    results.pages_per_sec_wall = total_pages / total_wall if total_wall > 0 else 0
    results.pages_per_sec_server = total_pages / total_server_time if total_server_time > 0 else 0
    results.docs_per_sec = len(host_paths) / total_wall if total_wall > 0 else 0
    results.jobs_submitted = len(batches)
    results.jobs_completed = completed
    results.jobs_failed = failed
    results.memory_before = mem_before or ""
    results.memory_after = mem_after or ""

    # Fix: memory peak by parsed numeric value instead of string comparison
    if mem_samples:
        peak_str, peak_val = max(mem_samples, key=lambda x: x[1])
        results.memory_peak = peak_str
    else:
        results.memory_peak = ""

    # Queue wait
    queue_waits = [jr.queue_wait for jr in job_results if jr.queue_wait > 0]
    results.avg_queue_wait = sum(queue_waits) / len(queue_waits) if queue_waits else 0.0

    # Client overhead
    overheads = [jr.client_overhead for jr in job_results if jr.client_overhead > 0]
    results.avg_client_overhead = sum(overheads) / len(overheads) if overheads else 0.0

    # Per-page latency stats
    all_page_times = sorted(pt for jr in job_results for pt in jr.page_times)
    if all_page_times:
        results.page_latency_p50 = percentile(all_page_times, 50)
        results.page_latency_p95 = percentile(all_page_times, 95)
        results.page_latency_p99 = percentile(all_page_times, 99)

    # Page errors
    results.total_page_errors = sum(jr.page_errors for jr in job_results)

    # Batch throughput variance
    if batch_throughputs:
        results.batch_throughput_mean = sum(batch_throughputs) / len(batch_throughputs)
        if len(batch_throughputs) > 1:
            mean = results.batch_throughput_mean
            variance = sum((x - mean) ** 2 for x in batch_throughputs) / (len(batch_throughputs) - 1)
            results.batch_throughput_std = math.sqrt(variance)

    # GPU utilization
    if gpu_samples:
        results.avg_gpu_utilization = sum(gpu_samples) / len(gpu_samples)
        results.gpu_samples = len(gpu_samples)

    # Render / OCR timing
    all_render_times = [rt for jr in job_results for rt in jr.page_render_times]
    all_ocr_times = [ot for jr in job_results for ot in jr.page_ocr_times]
    if all_render_times:
        results.avg_render_time = sum(all_render_times) / len(all_render_times)
    if all_ocr_times:
        results.avg_ocr_time = sum(all_ocr_times) / len(all_ocr_times)

    results.job_results = [asdict(jr) for jr in job_results]

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    if startup_time > 0:
        print(f"Startup time:     {startup_time:.1f}s")
    print(f"Sample set:       {sample_dir}/ ({len(host_paths)} docs"
          f"{', fixed' if existing_paths else ''})")
    print(f"Seed:             {args.seed}")
    print(f"PDF source:       {HOST_PDF_PATH}")
    if warmup_time > 0:
        print(f"Warmup:           {results.warmup_docs} docs "
              f"({warmup_batches} batch{'es' if warmup_batches > 1 else ''}), "
              f"{warmup_time:.1f}s")
    print(f"Documents:        {results.num_documents}")
    print(f"Total pages:      {results.total_pages}")
    print(f"Wall time:        {results.total_wall_time:.1f}s")
    print(f"Server time:      {results.total_server_time:.1f}s")
    print(f"Pages/sec (wall): {results.pages_per_sec_wall:.2f}")
    print(f"Pages/sec (svr):  {results.pages_per_sec_server:.2f}")
    print(f"Docs/sec:         {results.docs_per_sec:.2f}")
    print(f"Jobs:             {results.jobs_completed} completed, {results.jobs_failed} failed")
    print(f"Avg queue wait:   {results.avg_queue_wait:.2f}s")
    print(f"Client overhead:  {results.avg_client_overhead:.1f}s avg/job")
    if batch_throughputs:
        print(f"Batch throughput: {results.batch_throughput_mean:.1f} "
              f"\u00b1 {results.batch_throughput_std:.1f} pages/sec")
    if gpu_samples:
        print(f"GPU utilization:  {results.avg_gpu_utilization:.1f}% avg "
              f"({results.gpu_samples} samples)")
    if all_page_times:
        print(f"Page latency p50: {results.page_latency_p50:.2f}s")
        print(f"Page latency p95: {results.page_latency_p95:.2f}s")
        print(f"Page latency p99: {results.page_latency_p99:.2f}s")
    print(f"Page errors:      {results.total_page_errors} / {total_pages:,} pages")
    if all_render_times and all_ocr_times:
        total_page_time = results.avg_render_time + results.avg_ocr_time
        render_pct = (results.avg_render_time / total_page_time * 100) if total_page_time > 0 else 0
        ocr_pct = (results.avg_ocr_time / total_page_time * 100) if total_page_time > 0 else 0
        print(f"Avg render time:  {results.avg_render_time:.3f}s ({render_pct:.1f}% of page time)")
        print(f"Avg OCR time:     {results.avg_ocr_time:.3f}s ({ocr_pct:.1f}% of page time)")
    print(f"Memory before:    {results.memory_before}")
    print(f"Memory after:     {results.memory_after}")
    print(f"Memory peak:      {results.memory_peak}")

    # Server-reported delta
    pages_delta = (stats_after.get("total_pages_completed", 0) -
                   stats_before.get("total_pages_completed", 0))
    print(f"Server pages \u0394:   {pages_delta}")
    print(f"{'=' * 60}")

    # Save results
    if args.save:
        out_path = args.save
    else:
        out_path = f"docs/benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(results), f, indent=2)
    print(f"Results saved to: {out_path}")

    # Compare against baseline
    if args.compare:
        exit_code = print_comparison(args.compare, results, args.regression_threshold)
        return results, exit_code

    return results, 0


def fmt_delta(old: float, new: float, fmt: str = ".2f", suffix: str = "",
              invert: bool = False) -> str:
    """Format a metric comparison with delta percentage."""
    if old == 0 and new == 0:
        return f"{new:{fmt}}{suffix} (no change)"
    if old == 0:
        return f"{new:{fmt}}{suffix} (new)"
    pct = ((new - old) / abs(old)) * 100
    sign = "+" if pct >= 0 else ""
    arrow = "\u2192"
    return f"{old:{fmt}}{suffix} {arrow} {new:{fmt}}{suffix} ({sign}{pct:.1f}%)"


def print_comparison(baseline_path: str, current: BenchmarkResults,
                     regression_threshold: float) -> int:
    """Print comparison table and return exit code (1 if regression detected)."""
    try:
        with open(baseline_path) as f:
            baseline = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"\nERROR: Cannot load baseline from {baseline_path}: {e}")
        return 1

    b_ts = baseline.get("timestamp", "unknown")
    print(f"\nCOMPARISON vs baseline ({b_ts})")
    print("\u2500" * 55)

    rows = [
        ("Pages/sec (wall)", baseline.get("pages_per_sec_wall", 0),
         current.pages_per_sec_wall, ".1f", ""),
        ("Pages/sec (svr)", baseline.get("pages_per_sec_server", 0),
         current.pages_per_sec_server, ".1f", ""),
        ("Page p50", baseline.get("page_latency_p50", 0),
         current.page_latency_p50, ".2f", "s"),
        ("Page p95", baseline.get("page_latency_p95", 0),
         current.page_latency_p95, ".2f", "s"),
        ("Batch throughput \u03c3", baseline.get("batch_throughput_std", 0),
         current.batch_throughput_std, ".1f", ""),
        ("GPU utilization", baseline.get("avg_gpu_utilization", 0),
         current.avg_gpu_utilization, ".0f", "%"),
        ("Page errors", baseline.get("total_page_errors", 0),
         current.total_page_errors, ".0f", ""),
        ("Memory", parse_mem_usage(baseline.get("memory_peak", "")),
         parse_mem_usage(current.memory_peak), ".1f", "GiB"),
        ("Avg queue wait", baseline.get("avg_queue_wait", 0),
         current.avg_queue_wait, ".2f", "s"),
        ("Startup time", baseline.get("startup_time", 0),
         current.startup_time, ".1f", "s"),
        ("Avg render time", baseline.get("avg_render_time", 0),
         current.avg_render_time, ".3f", "s"),
        ("Avg OCR time", baseline.get("avg_ocr_time", 0),
         current.avg_ocr_time, ".3f", "s"),
    ]

    for label, old_val, new_val, fmt, suffix in rows:
        delta_str = fmt_delta(old_val, new_val, fmt, suffix)
        print(f"  {label + ':':<22s}{delta_str}")

    print()

    # Check for regression
    b_throughput = baseline.get("pages_per_sec_wall", 0)
    if b_throughput > 0:
        drop_pct = ((b_throughput - current.pages_per_sec_wall) / b_throughput) * 100
        if drop_pct > regression_threshold:
            print(f"REGRESSION DETECTED: throughput dropped {drop_pct:.1f}% "
                  f"(threshold: {regression_threshold}%)")
            return 1
        else:
            print(f"No regression: throughput delta within {regression_threshold}% threshold")

    return 0


def main():
    parser = argparse.ArgumentParser(description="OCR Server Benchmark")
    parser.add_argument("--docs", type=int, default=100,
                        help="Number of documents to benchmark (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Documents per batch (default: 16)")
    parser.add_argument("--max-concurrent", type=int, default=4,
                        help="Max concurrent batches (default: 4)")
    parser.add_argument("--server", default=SERVER_URL,
                        help=f"Server URL (default: {SERVER_URL})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for PDF sampling (default: 42)")
    parser.add_argument("--save", type=str, default=None,
                        help="Output file path (default: docs/benchmark_<timestamp>.json)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup batches before timed run (default: 1)")
    parser.add_argument("--sample-dir", type=str, default="benchmark_data",
                        help="Directory for fixed sample set (default: benchmark_data/)")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare against a previous benchmark JSON")
    parser.add_argument("--regression-threshold", type=float, default=10.0,
                        help="Fail if throughput drops more than this %% (default: 10)")
    parser.add_argument("--measure-startup", action="store_true",
                        help="Restart container and measure time to healthy")
    args = parser.parse_args()

    _, exit_code = asyncio.run(run_benchmark(args))
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
