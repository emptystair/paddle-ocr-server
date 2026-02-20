#!/usr/bin/env python3
"""Compare OCR accuracy between v4 TensorRT and paddle_v3 (PP-OCRv5 via native PaddleOCR 3.0).

v4 TRT uses PP-OCRv4 models with TensorRT acceleration (DPI=100).
paddle_v3 uses PP-OCRv5 models via native PaddleOCR 3.0 (DPI=100).

Usage:
    # 1. Make sure v4_trt is running on :8003 and paddle_v3 is STOPPED
    # 2. Run:
    python paddle_v4_tensorrt/compare_accuracy.py
"""

import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

# Same test corpus as the benchmark script — mix of eras
TEST_CORPUS = [
    # Old scanned docs (1988)
    "/data/1988/01/04/DOCC107906_004179-00684-1.pdf",
    "/data/1988/11/10/DOCC684702_004383-00743-1.pdf",
    "/data/1988/11/10/DOCC104040_004384-00084-1.pdf",
    "/data/1988/11/10/DOCC606938_004384-00462-1.pdf",
    "/data/1988/11/10/DOCC681707_004384-00425-1.pdf",
    "/data/1988/11/10/DOCC777993_004384-00258-1.pdf",
    "/data/1988/11/10/DOCC214616_004383-00665-1.pdf",
    "/data/1988/11/10/DOCC351933_004384-00065-1.pdf",
    "/data/1988/11/10/DOCC252613_BM1988-01598-1.pdf",
    "/data/1988/11/10/DOCC505773_004383-00732-1.pdf",
    # Medium age docs (2005)
    "/data/2005/11/10/DOCC107057675_011678-02614-1.pdf",
    "/data/2005/11/10/DOCC107058225_011679-02625-1.pdf",
    "/data/2005/11/10/DOCC107057503_BM2005-02227-1.pdf",
    "/data/2005/11/10/DOCC107057842_011679-00809-1.pdf",
    "/data/2005/11/10/DOCC107058215_011679-02359-1.pdf",
    # Modern docs (2024)
    "/data/2024/01/11/DOCC111700300_019517-00076-1.pdf",
    "/data/2024/01/11/DOCC111700002_019516-01026-1.pdf",
    "/data/2024/01/11/DOCC111700244_019516-02510-1.pdf",
    "/data/2024/01/11/DOCC111700262_019516-02471-1.pdf",
    "/data/2024/01/11/DOCC111700155_019516-01806-1.pdf",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V4_TRT_DIR = PROJECT_ROOT / "paddle_v4_tensorrt"
V5_DIR = PROJECT_ROOT / "paddle_v3" / "docker"


def run_cmd(cmd: str, cwd: str) -> str:
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return result.stdout + result.stderr


def docker_compose(action: str, cwd: str):
    """Run docker compose action."""
    print(f"  docker compose {action} in {Path(cwd).name}...")
    run_cmd(f"docker compose {action}", cwd)


def wait_for_server(url: str, max_wait: int = 120) -> bool:
    """Wait for server health endpoint."""
    import urllib.request
    print(f"  Waiting for {url}/health ...", end="", flush=True)
    for _ in range(max_wait):
        try:
            req = urllib.request.urlopen(f"{url}/health", timeout=2)
            if req.status == 200:
                print(" ready")
                return True
        except Exception:
            pass
        time.sleep(1)
        print(".", end="", flush=True)
    print(" TIMEOUT")
    return False


@dataclass
class DocResult:
    pdf_path: str
    pages: int = 0
    lines: int = 0
    chars: int = 0
    avg_confidence: float = 0.0
    confidences: List[float] = field(default_factory=list)
    full_text: str = ""
    page_texts: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


def extract_doc_result(results: Dict, pdf_path: str) -> DocResult:
    """Extract full text and metrics from OCR results for one document."""
    doc = DocResult(pdf_path=pdf_path)
    results_dict = results.get("results", {})

    doc_results = results_dict.get(pdf_path, {})
    if not doc_results:
        filename = pdf_path.split("/")[-1]
        for path, data in results_dict.items():
            if path.endswith(filename):
                doc_results = data
                break

    if not isinstance(doc_results, dict):
        doc.error = "No results found"
        return doc

    page_nums = sorted(doc_results.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    for page_num in page_nums:
        page_data = doc_results[page_num]
        if not isinstance(page_data, dict):
            continue
        doc.pages += 1
        page_lines = []
        for line in page_data.get("text_lines", []):
            if isinstance(line, dict):
                text = line.get("text", "")
                conf = line.get("confidence", 0)
                page_lines.append(text)
                doc.confidences.append(conf)
                doc.lines += 1
                doc.chars += len(text)
        doc.page_texts[page_num] = "\n".join(page_lines)

    doc.full_text = "\n\n".join(doc.page_texts.get(p, "") for p in page_nums)
    if doc.confidences:
        doc.avg_confidence = sum(doc.confidences) / len(doc.confidences)
    return doc


async def process_docs(server_url: str, docs: List[str], batch_size: int = 10,
                       timeout: float = 600.0) -> List[DocResult]:
    """Process documents through a server and return results."""
    all_results = []
    client_timeout = aiohttp.ClientTimeout(total=timeout + 30)

    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        for beg in range(0, len(docs), batch_size):
            batch = docs[beg:beg + batch_size]
            batch_num = beg // batch_size + 1
            total_batches = (len(docs) + batch_size - 1) // batch_size
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} docs)...", end="", flush=True)

            # Submit
            payload = {"pdf_paths": batch, "priority": 5}
            async with session.post(f"{server_url}/process", json=payload) as resp:
                if resp.status != 200:
                    print(f" SUBMIT ERROR {resp.status}")
                    for p in batch:
                        r = DocResult(pdf_path=p, error=f"Submit failed: {resp.status}")
                        all_results.append(r)
                    continue
                data = await resp.json()
                job_id = data.get("job_id")

            # Poll
            t0 = time.time()
            completed = {"completed", "completed_with_errors", "partial", "failed"}
            while time.time() - t0 < timeout:
                async with session.get(f"{server_url}/status/{job_id}") as resp:
                    status = (await resp.json()).get("status", "unknown")
                    if status in completed:
                        break
                await asyncio.sleep(0.5)
            else:
                print(f" TIMEOUT")
                for p in batch:
                    all_results.append(DocResult(pdf_path=p, error="Timeout"))
                continue

            # Get results
            async with session.get(f"{server_url}/results/{job_id}") as resp:
                results = await resp.json()

            elapsed = time.time() - t0
            total_pages = 0
            for p in batch:
                doc = extract_doc_result(results, p)
                all_results.append(doc)
                total_pages += doc.pages
            print(f" {total_pages} pages, {elapsed:.1f}s")

    return all_results


def compare_results(v4_results: List[DocResult], v5_results: List[DocResult]):
    """Print comparison report."""
    print()
    print("=" * 80)
    print("ACCURACY COMPARISON: v4 TensorRT (PP-OCRv4) vs paddle_v3 (PP-OCRv5)")
    print("=" * 80)

    # Build lookup
    v4_map = {r.pdf_path: r for r in v4_results}
    v5_map = {r.pdf_path: r for r in v5_results}

    # Categorize docs by era
    eras = {
        "1988 (old scans)": [p for p in TEST_CORPUS if "/1988/" in p],
        "2005 (medium age)": [p for p in TEST_CORPUS if "/2005/" in p],
        "2024 (modern)": [p for p in TEST_CORPUS if "/2024/" in p],
    }

    similarities = []

    # Per-document details
    print(f"\n{'Document':<45} {'v4 TRT':>20} {'v5 paddle_v3':>20} {'Similarity':>12}")
    print(f"{'':45} {'lines/chars/conf':>20} {'lines/chars/conf':>20} {'%':>12}")
    print("-" * 100)

    for era_name, era_paths in eras.items():
        print(f"\n  {era_name}:")
        for pdf_path in era_paths:
            short = pdf_path.split("/")[-1][:40]
            v4 = v4_map.get(pdf_path)
            v5 = v5_map.get(pdf_path)
            if not v4 or not v5:
                print(f"  {short:<43} {'MISSING':>20} {'MISSING':>20}")
                continue
            if v4.error or v5.error:
                print(f"  {short:<43} {v4.error or 'ok':>20} {v5.error or 'ok':>20}")
                continue

            v4_summary = f"{v4.lines}/{v4.chars}/{v4.avg_confidence:.2f}"
            v5_summary = f"{v5.lines}/{v5.chars}/{v5.avg_confidence:.2f}"

            # Text similarity
            sim = SequenceMatcher(None, v4.full_text, v5.full_text).ratio()
            similarities.append(sim)

            print(f"  {short:<43} {v4_summary:>20} {v5_summary:>20} {sim:>11.1%}")

    # Aggregates
    print()
    print("=" * 80)
    print("AGGREGATE SUMMARY")
    print("=" * 80)

    v4_total_lines = sum(r.lines for r in v4_results if not r.error)
    v5_total_lines = sum(r.lines for r in v5_results if not r.error)
    v4_total_chars = sum(r.chars for r in v4_results if not r.error)
    v5_total_chars = sum(r.chars for r in v5_results if not r.error)
    v4_confs = [c for r in v4_results if not r.error for c in r.confidences]
    v5_confs = [c for r in v5_results if not r.error for c in r.confidences]

    print(f"\n{'Metric':<30} {'v4 TRT':>15} {'v5 paddle_v3':>15} {'Diff':>10}")
    print("-" * 72)
    print(f"{'Total lines':<30} {v4_total_lines:>15,} {v5_total_lines:>15,} {v4_total_lines - v5_total_lines:>+10,}")
    print(f"{'Total chars':<30} {v4_total_chars:>15,} {v5_total_chars:>15,} {v4_total_chars - v5_total_chars:>+10,}")

    v4_avg = 0
    if v4_confs:
        v4_avg = sum(v4_confs) / len(v4_confs)
        print(f"{'Avg confidence':<30} {v4_avg:>15.4f}", end="")
    else:
        print(f"{'Avg confidence':<30} {'N/A':>15}", end="")
    if v5_confs:
        v5_avg = sum(v5_confs) / len(v5_confs)
        print(f" {v5_avg:>15.4f} {v4_avg - v5_avg:>+10.4f}")
    else:
        print(f" {'N/A':>15}")

    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        min_sim = min(similarities)
        max_sim = max(similarities)
        print(f"\n{'Avg text similarity':<30} {avg_sim:>15.1%}")
        print(f"{'Min text similarity':<30} {min_sim:>15.1%}")
        print(f"{'Max text similarity':<30} {max_sim:>15.1%}")

    # Per-era summary
    print(f"\n{'Era':<25} {'v4 avg conf':>12} {'v5 avg conf':>12} {'Avg similarity':>15}")
    print("-" * 66)
    for era_name, era_paths in eras.items():
        era_v4_confs = [c for p in era_paths for c in v4_map.get(p, DocResult(p)).confidences]
        era_v5_confs = [c for p in era_paths for c in v5_map.get(p, DocResult(p)).confidences]
        era_sims = []
        for p in era_paths:
            v4 = v4_map.get(p)
            v5 = v5_map.get(p)
            if v4 and v5 and not v4.error and not v5.error:
                era_sims.append(SequenceMatcher(None, v4.full_text, v5.full_text).ratio())

        v4_avg_c = sum(era_v4_confs) / len(era_v4_confs) if era_v4_confs else 0
        v5_avg_c = sum(era_v5_confs) / len(era_v5_confs) if era_v5_confs else 0
        avg_s = sum(era_sims) / len(era_sims) if era_sims else 0
        print(f"{era_name:<25} {v4_avg_c:>12.4f} {v5_avg_c:>12.4f} {avg_s:>14.1%}")

    print()

    # Save raw results
    output = {
        "timestamp": datetime.now().isoformat(),
        "docs": len(TEST_CORPUS),
        "v4_trt": [{"pdf": r.pdf_path, "lines": r.lines, "chars": r.chars,
                     "avg_conf": r.avg_confidence, "pages": r.pages,
                     "text": r.full_text, "error": r.error} for r in v4_results],
        "v5_paddle_v3": [{"pdf": r.pdf_path, "lines": r.lines, "chars": r.chars,
                    "avg_conf": r.avg_confidence, "pages": r.pages,
                    "text": r.full_text, "error": r.error} for r in v5_results],
    }
    out_path = PROJECT_ROOT / f"accuracy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Raw results saved to: {out_path.name}")


async def main():
    print("=" * 80)
    print("OCR Accuracy Comparison: v4 TensorRT (PP-OCRv4) vs paddle_v3 (PP-OCRv5)")
    print(f"Documents: {len(TEST_CORPUS)}")
    print("v4 TRT: PP-OCRv4 + ONNX RT TensorRT (DPI=100) | v5: PP-OCRv5 native PaddleOCR 3.0 (DPI=100)")
    print("=" * 80)

    # Phase 1: Process through v4 TRT (should be running on :8003)
    print("\n--- Phase 1: Processing through v4 TensorRT (port 8003) ---")
    if not wait_for_server("http://localhost:8003"):
        print("ERROR: v4 TRT server not available on :8003")
        sys.exit(1)
    v4_results = await process_docs("http://localhost:8003", TEST_CORPUS)
    v4_ok = sum(1 for r in v4_results if not r.error)
    print(f"  Completed: {v4_ok}/{len(TEST_CORPUS)} docs")

    # Phase 2: Stop v4 TRT, start paddle_v3
    print("\n--- Phase 2: Switching to paddle_v3 (PP-OCRv5) ---")
    docker_compose("down", str(V4_TRT_DIR))
    print("  Waiting 10s for GPU memory release...")
    await asyncio.sleep(10)
    docker_compose("up -d ocr-server-1", str(V5_DIR))
    if not wait_for_server("http://localhost:8000", max_wait=180):
        print("ERROR: paddle_v3 server not available on :8000")
        sys.exit(1)
    # Wait for workers to initialize
    print("  Waiting 30s for workers to initialize...")
    await asyncio.sleep(30)

    # Phase 3: Process through paddle_v3
    print("\n--- Phase 3: Processing through paddle_v3 / PP-OCRv5 (port 8000) ---")
    v5_results = await process_docs("http://localhost:8000", TEST_CORPUS)
    v5_ok = sum(1 for r in v5_results if not r.error)
    print(f"  Completed: {v5_ok}/{len(TEST_CORPUS)} docs")

    # Phase 4: Restore — stop paddle_v3, restart v4 TRT
    print("\n--- Phase 4: Restoring v4 TRT server ---")
    docker_compose("down", str(V5_DIR))
    print("  Waiting 10s for GPU memory release...")
    await asyncio.sleep(10)
    docker_compose("up -d", str(V4_TRT_DIR))

    # Phase 5: Compare
    compare_results(v4_results, v5_results)


if __name__ == "__main__":
    asyncio.run(main())
