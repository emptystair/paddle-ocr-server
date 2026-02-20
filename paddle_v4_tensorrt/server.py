#!/usr/bin/env python3
"""
PP-OCRv4 + TensorRT via ONNX Runtime Server

Direct ONNX Runtime sessions with TensorRT Execution Provider — no RapidOCR wrapper.
Reimplements the detection → classification → recognition pipeline from scratch.
"""

import os
import asyncio
import math
import time
import uuid
import numpy as np
import cv2
import pypdfium2 as pdfium
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import multiprocessing as mp
from multiprocessing import shared_memory
import threading
import queue
from loguru import logger
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psutil
import GPUtil
from pyclipper import PyclipperOffset, JT_ROUND, ET_CLOSEDPOLYGON
from shapely.geometry import Polygon

# Configure logger
logger.add("page_ocr_v4_trt.log", rotation="100 MB", retention="7 days", level="INFO")

# Server configuration
CONFIG = {
    "NUM_WORKERS": int(os.getenv("NUM_WORKERS", "4")),
    "DPI": int(os.getenv("DPI", "100")),
    "PAGE_QUEUE_SIZE": int(os.getenv("PAGE_QUEUE_SIZE", "100")),
    "BUFFER_POOL_SIZE": int(os.getenv("BUFFER_POOL_SIZE", "50")),
    "MAX_PAGE_WIDTH": int(os.getenv("MAX_PAGE_WIDTH", "2200")),
    "MAX_PAGE_HEIGHT": int(os.getenv("MAX_PAGE_HEIGHT", "2800")),
    "SERVER_PORT": int(os.getenv("SERVER_PORT", "8000")),
    "INSTANCE_ID": os.getenv("INSTANCE_ID", "v4_tensorrt"),
    "DET_DB_THRESH": float(os.getenv("DET_DB_THRESH", "0.3")),
    "DET_DB_BOX_THRESH": float(os.getenv("DET_DB_BOX_THRESH", "0.5")),
    "DET_DB_UNCLIP_RATIO": float(os.getenv("DET_DB_UNCLIP_RATIO", "1.6")),
    "PDF_CACHE_SIZE": int(os.getenv("PDF_CACHE_SIZE", "50")),
    "TEXT_SCORE": float(os.getenv("TEXT_SCORE", "0.3")),
    "CLS_THRESH": float(os.getenv("CLS_THRESH", "0.9")),
    "REC_BATCH_SIZE": int(os.getenv("REC_BATCH_SIZE", "6")),
    "DET_MODEL_PATH": os.getenv("DET_MODEL_PATH", "/app/models/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx"),
    "REC_MODEL_PATH": os.getenv("REC_MODEL_PATH", "/app/models/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx"),
    "CLS_MODEL_PATH": os.getenv("CLS_MODEL_PATH", "/app/models/PP-OCRv4/ch_ppocr_mobile_v2.0_cls_infer.onnx"),
    "REC_KEYS_PATH": os.getenv("REC_KEYS_PATH", "/app/models/PP-OCRv4/ppocr_keys_v1.txt"),
}

BUFFER_SIZE = CONFIG["MAX_PAGE_WIDTH"] * CONFIG["MAX_PAGE_HEIGHT"] * 3

# Global state
page_queue = None
result_queue = None
job_manager = None
buffer_manager = None

app = FastAPI(title=f"PP-OCRv4 TensorRT Server - {CONFIG['INSTANCE_ID']}", version="2.0.0")


class OCRRequest(BaseModel):
    pdf_paths: List[str]
    priority: int = 5
    dpi: Optional[int] = None
    callback_url: Optional[str] = None


class OCRResponse(BaseModel):
    job_id: str
    status: str
    message: str


@dataclass
class PageTask:
    job_id: str
    pdf_path: str
    page_num: int
    total_pages: int
    buffer_name: str
    img_shape: Tuple[int, int, int]
    priority: int
    submit_time: float
    dpi: Optional[int] = None


@dataclass
class EnqueueRequest:
    job_id: str
    pdf_paths: List[str]
    priority: int
    dpi: Optional[int] = None


dispatch_queue = queue.Queue()
dispatcher_thread = None
dispatcher_stop = threading.Event()


# ---------------------------------------------------------------------------
# OCR Pipeline — direct ONNX Runtime with TensorRT EP
# ---------------------------------------------------------------------------

def create_ort_session(model_path: str, use_trt: bool = True) -> ort.InferenceSession:
    """Create an ONNX Runtime session with TensorRT → CUDA → CPU fallback."""
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # Warning level
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = []
    available = ort.get_available_providers()

    if use_trt and "TensorrtExecutionProvider" in available:
        trt_opts = {
            "device_id": 0,
            "trt_max_workspace_size": str(4 * 1024 * 1024 * 1024),  # 4 GB
            "trt_fp16_enable": "True",
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": os.getenv("TRT_ENGINE_CACHE_PATH", "/app/trt_cache"),
        }
        providers.append(("TensorrtExecutionProvider", trt_opts))

    if "CUDAExecutionProvider" in available:
        cuda_opts = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "DEFAULT",
            "do_copy_in_default_stream": True,
        }
        providers.append(("CUDAExecutionProvider", cuda_opts))

    providers.append(("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))

    session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
    active = session.get_providers()
    logger.info(f"Session for {Path(model_path).name}: active providers = {active}")
    return session


def load_character_list(keys_path: str) -> List[str]:
    """Load character dictionary for CTC decoding."""
    with open(keys_path, "r", encoding="utf-8") as f:
        chars = [line.strip("\n") for line in f]
    # Index 0 = CTC blank, then characters, then space at end
    chars = ["blank"] + chars + [" "]
    return chars


# -- Detection pre/post-processing --

def det_preprocess(img: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Resize, normalize, transpose for detection model."""
    h, w = img.shape[:2]

    # Fixed limit_side_len to match paddle_v3
    limit = int(os.getenv("DET_LIMIT_SIDE_LEN", "1920"))
    max_side = max(h, w)

    ratio = 1.0
    if max_side > limit:
        ratio = limit / max_side

    resize_h = max(int(round(h * ratio / 32) * 32), 32)
    resize_w = max(int(round(w * ratio / 32) * 32), 32)

    resized = cv2.resize(img, (resize_w, resize_h))
    # ImageNet normalization (PaddleOCR det standard)
    normed = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normed = (normed - mean) / std
    # HWC → CHW, add batch
    tensor = normed.transpose((2, 0, 1))[np.newaxis, ...]
    return tensor, h / resize_h, w / resize_w


def _get_mini_boxes(contour):
    """Get minimum bounding rectangle as 4 sorted points + shortest side."""
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2
    box = np.array([points[index_1], points[index_2], points[index_3], points[index_4]])
    return box, min(bounding_box[1])


def _box_score_fast(pred: np.ndarray, box: np.ndarray) -> float:
    """Calculate average score inside box region."""
    h, w = pred.shape[:2]
    box_ = box.copy()
    xmin = int(np.clip(np.floor(box_[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(box_[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(box_[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(box_[:, 1].max()), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box_[:, 0] -= xmin
    box_[:, 1] -= ymin
    cv2.fillPoly(mask, box_.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(pred[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def _unclip(box: np.ndarray, unclip_ratio: float) -> np.ndarray:
    """Expand polygon using Clipper library."""
    poly = Polygon(box)
    if poly.area == 0:
        return box
    distance = poly.area * unclip_ratio / poly.length
    offset = PyclipperOffset()
    offset.AddPath(box.astype(int).tolist(), JT_ROUND, ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    if not expanded:
        return box
    return np.array(expanded[0])


def det_postprocess(pred: np.ndarray, src_h: int, src_w: int, config: dict) -> List[np.ndarray]:
    """DB post-processing: binarize → contours → unclip → scale to original."""
    thresh = config["DET_DB_THRESH"]
    box_thresh = config["DET_DB_BOX_THRESH"]
    unclip_ratio = config["DET_DB_UNCLIP_RATIO"]

    pred_map = pred[0, 0]  # (H, W)
    bitmap = (pred_map > thresh).astype(np.uint8)

    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    det_h, det_w = pred_map.shape

    for contour in contours[:1000]:
        points, sside = _get_mini_boxes(contour)
        if sside < 3:
            continue
        score = _box_score_fast(pred_map, points.reshape(-1, 2))
        if score < box_thresh:
            continue

        expanded = _unclip(points, unclip_ratio)
        expanded_contour = expanded.reshape(-1, 1, 2).astype(np.int32)
        box, sside = _get_mini_boxes(expanded_contour)
        if sside < 5:
            continue

        box[:, 0] = np.clip(np.round(box[:, 0] / det_w * src_w), 0, src_w)
        box[:, 1] = np.clip(np.round(box[:, 1] / det_h * src_h), 0, src_h)
        boxes.append(box.astype(np.int32))

    return boxes


# -- Crop text regions --

def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Perspective-transform a 4-point box into a horizontal strip."""
    crop_w = int(max(np.linalg.norm(points[0] - points[1]),
                     np.linalg.norm(points[2] - points[3])))
    crop_h = int(max(np.linalg.norm(points[0] - points[3]),
                     np.linalg.norm(points[1] - points[2])))
    if crop_w == 0 or crop_h == 0:
        return None
    pts_std = np.array([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
    dst = cv2.warpPerspective(img, M, (crop_w, crop_h),
                              borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if dst.shape[0] * 1.0 / dst.shape[1] >= 1.5:
        dst = np.rot90(dst)
    return dst


# -- Classification pre/post-processing --

def cls_preprocess(img: np.ndarray) -> np.ndarray:
    """Resize and normalize for classifier. Returns (1, 3, 48, 192)."""
    img_h, img_w = 48, 192
    h, w = img.shape[:2]
    ratio = w / float(h)
    resized_w = min(int(math.ceil(img_h * ratio)), img_w)
    resized = cv2.resize(img, (resized_w, img_h)).astype(np.float32)
    resized = resized.transpose((2, 0, 1)) / 127.5 - 1.0
    padded = np.zeros((3, img_h, img_w), dtype=np.float32)
    padded[:, :, :resized_w] = resized
    return padded[np.newaxis, ...]


# -- Recognition pre/post-processing --

def rec_preprocess(img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
    """Resize and normalize for recognizer. Returns (1, 3, 48, W).

    Width is quantized to the nearest multiple of 128 (capped at 3200) so that
    TensorRT only needs to compile a small fixed set of engines (25 widths)
    instead of one per unique pixel width.
    """
    img_h = 48
    img_w = int(img_h * max_wh_ratio)
    # Quantize to nearest multiple of 128, min 128, max 3200
    img_w = min(max(((img_w + 127) // 128) * 128, 128), 3200)
    h, w = img.shape[:2]
    ratio = w / float(h)
    resized_w = min(int(math.ceil(img_h * ratio)), img_w)
    resized = cv2.resize(img, (resized_w, img_h)).astype(np.float32)
    resized = resized.transpose((2, 0, 1)) / 127.5 - 1.0
    padded = np.zeros((3, img_h, img_w), dtype=np.float32)
    padded[:, :, :resized_w] = resized
    return padded


def ctc_decode(preds: np.ndarray, char_list: List[str]) -> List[Tuple[str, float]]:
    """CTC greedy decode: argmax → remove duplicates & blanks."""
    pred_idxs = preds.argmax(axis=2)   # (B, T)
    pred_probs = preds.max(axis=2)     # (B, T)
    results = []
    for b in range(pred_idxs.shape[0]):
        idxs = pred_idxs[b]
        probs = pred_probs[b]
        # Remove duplicates
        selection = np.ones(len(idxs), dtype=bool)
        selection[1:] = idxs[1:] != idxs[:-1]
        # Remove blank (index 0)
        selection &= idxs != 0
        chars = [char_list[i] for i in idxs[selection]]
        conf_list = probs[selection]
        text = "".join(chars)
        conf = float(np.mean(conf_list)) if len(conf_list) > 0 else 0.0
        results.append((text, conf))
    return results


# Common detection input shapes from Wake County dataset at DPI=100.
# Covers ~99.9% of pages. Each shape triggers a TRT engine compilation
# on first encounter (~1-3 min each), so we warm them up at startup.
WARMUP_DET_SHAPES = [
    (1408, 832),   # 69.0% of pages
    (1408, 864),   # 26.4%
    (1792, 2400),  #  2.2%
    (1984, 2400),  #  0.7%
    (1440, 896),   #  0.7%
    (1408, 896),   #  0.7%
    (1440, 928),   #  0.1%
    (2400, 1792),  #  0.1%
    (1088, 832),   #  0.1%
    (1440, 960),   #  0.1%
]

# Recognition widths quantized to multiples of 64 (matching rec_preprocess).
# Warm up ALL possible bucket widths so TRT engines are compiled once at startup.
REC_WIDTH_STEP = int(os.getenv("REC_WIDTH_STEP", "128"))
WARMUP_REC_WIDTHS = list(range(REC_WIDTH_STEP, 3200 + 1, REC_WIDTH_STEP))

# Detection input shapes to pre-cache for TRT.
# Profiled from 3.47M PDFs — top 50 shapes cover 99.7% of all pages.
# Format: (H, W) — these are the resized dimensions after det_preprocess.
WARMUP_DET_SHAPES = [
    # Original 20 shapes (modern documents, 95% of corpus)
    (1408, 832),   # 35.9%
    (1408, 864),   # 30.5%
    (1088, 832),   # 9.8%
    (1376, 832),   # 7.1%
    (1088, 864),   # 3.9%
    (1376, 864),   # 3.3%
    (1472, 1920),  # 1.5%
    (1120, 832),   # 1.4%
    (1504, 1920),  # 1.3%
    (1120, 864),   # 1.2%
    (1536, 1920),  # 1.0%
    (1408, 896),   # 0.8%
    (1440, 896),   # 0.7%
    (1600, 1920),  # 0.5%
    (1440, 1920),  # 0.4%
    (1440, 928),   # 0.3%
    (1472, 960),   # 0.2%
    (1152, 832),   # 0.1%
    (1504, 1024),  # 0.1%
    (1568, 1920),  # 0.1%
    # Additional 30 shapes (pre-2000 documents, oversized/non-standard pages)
    (1088, 896),   # 45.5K pages
    (1632, 1056),  # 44.7K pages
    (1440, 864),   # 37.8K pages
    (1632, 1024),  # 34.0K pages
    (1472, 992),   # 32.8K pages
    (1632, 1152),  # 25.1K pages
    (1504, 928),   # 19.3K pages
    (1600, 928),   # 17.3K pages
    (1568, 960),   # 16.8K pages
    (1568, 928),   # 16.6K pages
    (1792, 1152),  # 13.5K pages
    (1152, 864),   # 7.0K pages
    (1120, 896),   # 6.3K pages
    (1696, 1088),  # 5.4K pages
    (1696, 1184),  # 5.1K pages
    (1760, 1120),  # 4.5K pages
    (1504, 960),   # 3.5K pages
    (1504, 992),   # 3.4K pages
    (1792, 1184),  # 2.6K pages
    (1280, 1920),  # 2.5K pages
    (1152, 896),   # 2.4K pages
    (1792, 1120),  # 2.4K pages
    (1472, 928),   # 2.3K pages
    (1216, 960),   # 2.0K pages
    (1600, 992),   # 1.9K pages
    (1056, 832),   # 1.8K pages
    (1472, 896),   # 1.8K pages
    (1184, 832),   # 1.7K pages
    (1440, 960),   # 1.5K pages
    (1440, 832),   # 1.3K pages
]

# Set of all warmed-up shapes for fast lookup in workers.
# Pages with det input shapes not in this set are skipped to avoid
# multi-minute TRT engine compilation stalling the pipeline.
CACHED_DET_SHAPES = set(WARMUP_DET_SHAPES)


def det_input_shape(img_h: int, img_w: int) -> Tuple[int, int]:
    """Compute the det model input shape for a given image, matching det_preprocess logic."""
    limit = int(os.getenv("DET_LIMIT_SIDE_LEN", "1920"))
    max_side = max(img_h, img_w)
    ratio = 1.0
    if max_side > limit:
        ratio = limit / max_side
    resize_h = max(int(round(img_h * ratio / 32) * 32), 32)
    resize_w = max(int(round(img_w * ratio / 32) * 32), 32)
    return (resize_h, resize_w)


def warmup_trt_engines(config: dict):
    """Pre-compile TRT engines for common input shapes.

    Runs in the main process BEFORE workers start, so engines are cached
    and workers load them instantly without GPU contention.
    """
    cache_path = os.getenv("TRT_ENGINE_CACHE_PATH", "/app/trt_cache")
    existing = set(os.listdir(cache_path)) if os.path.isdir(cache_path) else set()
    if existing:
        engine_count = sum(1 for f in existing if f.endswith(".engine"))
        logger.info(f"TRT cache already has {engine_count} engines — checking if warmup needed")

    logger.info("=== TRT Engine Warmup: det + rec models ===")
    t_start = time.time()

    # Warm up detection model for common page shapes
    det_session = create_ort_session(config["DET_MODEL_PATH"], use_trt=True)
    det_input = det_session.get_inputs()[0].name

    for h, w in WARMUP_DET_SHAPES:
        logger.info(f"Warming up DET model (1, 3, {h}, {w})...")
        dummy_det = np.random.randn(1, 3, h, w).astype(np.float32)
        det_session.run(None, {det_input: dummy_det})
        logger.info(f"  DET ({h}x{w}) warmup done")

    del det_session

    # Warm up recognition model for all quantized widths
    rec_session = create_ort_session(config["REC_MODEL_PATH"], use_trt=True)
    rec_input = rec_session.get_inputs()[0].name

    rec_batch_size = config.get("REC_BATCH_SIZE", 1)
    warmup_batches = [1]
    if rec_batch_size > 1:
        warmup_batches.append(rec_batch_size)
    for bs in warmup_batches:
        for w in WARMUP_REC_WIDTHS:
            logger.info(f"Warming up REC model ({bs}, 3, 48, {w})...")
            dummy_rec = np.random.randn(bs, 3, 48, w).astype(np.float32)
            rec_session.run(None, {rec_input: dummy_rec})
            logger.info(f"  REC ({bs}x{w}) warmup done")

    elapsed = time.time() - t_start
    new_engines = set(os.listdir(cache_path)) if os.path.isdir(cache_path) else set()
    new_count = sum(1 for f in (new_engines - existing) if f.endswith(".engine"))
    logger.info(f"=== TRT Warmup complete: {new_count} new engines in {elapsed:.1f}s ===")

    del rec_session


class OCREngine:
    """Direct ONNX Runtime OCR pipeline with TensorRT EP."""

    def __init__(self, config: dict):
        self.config = config
        logger.info("Creating sessions: det=TRT+CUDA, cls=CUDA, rec=TRT+CUDA...")
        self.det_session = create_ort_session(config["DET_MODEL_PATH"], use_trt=True)
        self.cls_session = create_ort_session(config["CLS_MODEL_PATH"], use_trt=False)
        self.rec_session = create_ort_session(config["REC_MODEL_PATH"], use_trt=True)

        self.det_input_name = self.det_session.get_inputs()[0].name
        self.cls_input_name = self.cls_session.get_inputs()[0].name
        self.rec_input_name = self.rec_session.get_inputs()[0].name

        self.char_list = load_character_list(config["REC_KEYS_PATH"])
        logger.info(f"Loaded {len(self.char_list)} characters for CTC decoding")

    def __call__(self, img: np.ndarray):
        """Run full OCR pipeline. Returns list of [bbox, text, confidence]."""
        t0 = time.time()
        src_h, src_w = img.shape[:2]

        # 1. Detection
        det_input, ratio_h, ratio_w = det_preprocess(img)
        det_output = self.det_session.run(None, {self.det_input_name: det_input})[0]
        boxes = det_postprocess(det_output, src_h, src_w, self.config)

        if not boxes:
            return [], time.time() - t0

        # Sort boxes top-to-bottom, left-to-right
        boxes.sort(key=lambda b: (b[0][1], b[0][0]))

        # 2. Crop text regions
        crop_imgs = []
        valid_boxes = []
        for box in boxes:
            crop = get_rotate_crop_image(img, box)
            if crop is not None and crop.size > 0:
                crop_imgs.append(crop)
                valid_boxes.append(box)

        if not crop_imgs:
            return [], time.time() - t0

        # 3. Classification (orientation check) — DISABLED
        # paddle_v3 uses use_textline_orientation=False and achieves 99.1%.
        # The cls model is unreliable on thin crops at low DPI and can
        # incorrectly flip right-side-up text, causing worse results.

        # 4. Recognition (batched by quantized width)
        # Group crops by their quantized target width so all crops in a batch
        # have the same padded dimensions — critical for TRT accuracy.
        batch_size = self.config["REC_BATCH_SIZE"]
        rec_results = [None] * len(crop_imgs)

        # Compute quantized width for each crop
        img_h = 48
        width_groups = {}  # quantized_width -> list of (idx, crop)
        for i, crop in enumerate(crop_imgs):
            h, w = crop.shape[:2]
            wh_ratio = max(w / max(h, 1), 320.0 / 48.0)
            target_w = int(img_h * wh_ratio)
            step = REC_WIDTH_STEP
            quant_w = min(max(((target_w + step - 1) // step) * step, step), 3200)
            if quant_w not in width_groups:
                width_groups[quant_w] = []
            width_groups[quant_w].append(i)

        for quant_w, group_indices in width_groups.items():
            fixed_ratio = quant_w / img_h
            for beg in range(0, len(group_indices), batch_size):
                end = min(len(group_indices), beg + batch_size)
                batch_indices = group_indices[beg:end]

                batch = []
                for idx in batch_indices:
                    normed = rec_preprocess(crop_imgs[idx], fixed_ratio)
                    batch.append(normed[np.newaxis, ...])
                batch_tensor = np.concatenate(batch).astype(np.float32)

                preds = self.rec_session.run(None, {self.rec_input_name: batch_tensor})[0]
                decoded = ctc_decode(preds, self.char_list)

                for j, idx in enumerate(batch_indices):
                    rec_results[idx] = decoded[j]

        # 5. Assemble results
        text_score = self.config["TEXT_SCORE"]
        results = []
        for i, (text, conf) in enumerate(rec_results):
            if text and conf >= text_score:
                results.append([valid_boxes[i], text, conf])
        return results, time.time() - t0


# ---------------------------------------------------------------------------
# Shared memory buffer pool (unchanged)
# ---------------------------------------------------------------------------

class SharedBufferPool:
    def __init__(self, pool_size: int, buffer_size: int, instance_id: str):
        self.pool_size = pool_size
        self.buffer_size = buffer_size
        self.instance_id = instance_id
        self.buffers = {}
        self.available = mp.Queue()

        for i in range(pool_size):
            name = f"ocr_buffer_{instance_id}_{i}"
            try:
                shm = shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except:
                pass
            shm = shared_memory.SharedMemory(create=True, size=buffer_size, name=name)
            self.buffers[name] = shm
            self.available.put(name)
        logger.info(f"[{instance_id}] Initialized {pool_size} buffers of {buffer_size/1024/1024:.1f}MB each")

    def get_buffer(self, timeout=10):
        try:
            return self.available.get(timeout=timeout)
        except:
            return None

    def return_buffer(self, name):
        if name in self.buffers:
            self.available.put(name)

    def cleanup(self):
        for _, shm in self.buffers.items():
            try:
                shm.close()
                shm.unlink()
            except:
                pass


# ---------------------------------------------------------------------------
# PDF Renderer (unchanged)
# ---------------------------------------------------------------------------

class PDFRenderer:
    def __init__(self, dpi: int, buffer_pool: SharedBufferPool, pdf_cache_size: int = 10):
        self.dpi = dpi
        self.buffer_pool = buffer_pool
        self.pdf_cache_size = pdf_cache_size
        self.pdf_cache = OrderedDict()

    def _get_pdf(self, pdf_path: str):
        if pdf_path in self.pdf_cache:
            self.pdf_cache.move_to_end(pdf_path)
            return self.pdf_cache[pdf_path]
        pdf = pdfium.PdfDocument(pdf_path)
        self.pdf_cache[pdf_path] = pdf
        if len(self.pdf_cache) > self.pdf_cache_size:
            oldest_path = next(iter(self.pdf_cache))
            try:
                self.pdf_cache[oldest_path].close()
            except:
                pass
            del self.pdf_cache[oldest_path]
        return pdf

    def cleanup(self):
        for _, pdf in list(self.pdf_cache.items()):
            try:
                pdf.close()
            except:
                pass
        self.pdf_cache.clear()

    def render_page_to_buffer(self, pdf_path: str, page_num: int, buffer_name: str, dpi: int = None):
        try:
            render_dpi = dpi or self.dpi
            pdf = self._get_pdf(pdf_path)
            if page_num >= len(pdf):
                return None
            page = pdf[page_num]
            scale = render_dpi / 72
            width = int(page.get_width() * scale)
            height = int(page.get_height() * scale)
            required = height * width * 3
            if required > self.buffer_pool.buffer_size:
                fit = min(CONFIG["MAX_PAGE_WIDTH"] / width, CONFIG["MAX_PAGE_HEIGHT"] / height, 1.0)
                scale *= fit
                width = int(page.get_width() * scale)
                height = int(page.get_height() * scale)
            shm = shared_memory.SharedMemory(name=buffer_name)
            buf = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)
            bitmap = page.render(scale=scale, rotation=0)
            np_img = bitmap.to_numpy()[:, :, :3]
            mh, mw = min(np_img.shape[0], buf.shape[0]), min(np_img.shape[1], buf.shape[1])
            buf[:mh, :mw, :] = np_img[:mh, :mw, :]
            shm.close()
            return (height, width, 3)
        except Exception as e:
            logger.error(f"Render failed page {page_num} of {pdf_path}: {e}")
            return None


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

class PageOCRWorker(mp.Process):
    def __init__(self, worker_id, task_queue, result_queue, buffer_pool, config, gpu_id):
        super().__init__()
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.buffer_pool = buffer_pool
        self.config = config
        self.gpu_id = gpu_id

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        logger.info(f"Worker {self.worker_id} initializing OCR engine (GPU {self.gpu_id})...")
        engine = OCREngine(self.config)
        renderer = PDFRenderer(self.config["DPI"], self.buffer_pool, self.config["PDF_CACHE_SIZE"])

        # Per-worker warmup: run dummy inference to trigger cuDNN autotuning
        t_warm = time.time()
        dummy_det = np.random.randn(1, 3, 1408, 832).astype(np.float32)
        engine.det_session.run(None, {engine.det_input_name: dummy_det})
        dummy_cls = np.random.randn(1, 3, 48, 192).astype(np.float32)
        engine.cls_session.run(None, {engine.cls_input_name: dummy_cls})
        for w in [320, 480, 640, 960]:
            dummy_rec = np.random.randn(1, 3, 48, w).astype(np.float32)
            engine.rec_session.run(None, {engine.rec_input_name: dummy_rec})
        logger.info(f"Worker {self.worker_id} ready (warmup {time.time()-t_warm:.1f}s)")

        try:
            while True:
                try:
                    task = self.task_queue.get(timeout=1)
                    if task is None:
                        break
                    result = self._process(task, engine, renderer)
                    self.result_queue.put(result)
                    self.buffer_pool.return_buffer(task.buffer_name)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker {self.worker_id} error: {e}")
        finally:
            renderer.cleanup()
            logger.info(f"Worker {self.worker_id} stopped")

    def _process(self, task: PageTask, engine: OCREngine, renderer: PDFRenderer):
        t0 = time.time()
        try:
            shape = renderer.render_page_to_buffer(task.pdf_path, task.page_num, task.buffer_name, task.dpi)
            t_render = time.time() - t0
            if shape is None:
                return {"job_id": task.job_id, "page_num": task.page_num,
                        "pdf_path": task.pdf_path, "status": "error", "error": "Render failed"}
            # Check if det input shape is cached to avoid TRT recompilation stalls
            det_shape = det_input_shape(shape[0], shape[1])
            if det_shape not in CACHED_DET_SHAPES:
                logger.warning(f"W{self.worker_id} pg{task.page_num}: skipping uncached det shape {det_shape} for {task.pdf_path}")
                return {"job_id": task.job_id, "page_num": task.page_num,
                        "pdf_path": task.pdf_path, "status": "error",
                        "error": f"Uncached det shape {det_shape}"}

            shm = shared_memory.SharedMemory(name=task.buffer_name)
            img = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf).copy()
            shm.close()

            t_ocr_start = time.time()
            ocr_result, _ = engine(img)
            text_lines = []
            for line in ocr_result:
                bbox, text, conf = line
                if hasattr(bbox, "tolist"):
                    bbox = bbox.tolist()
                text_lines.append({"text": text, "confidence": float(conf), "bbox": bbox})

            t_ocr = time.time() - t_ocr_start
            logger.debug(f"W{self.worker_id} pg{task.page_num}: render={t_render:.3f}s ocr={t_ocr:.3f}s lines={len(text_lines)}")
            return {"job_id": task.job_id, "page_num": task.page_num, "pdf_path": task.pdf_path,
                    "status": "success", "text_lines": text_lines, "process_time": time.time() - t0}
        except Exception as e:
            logger.error(f"Process page {task.page_num} failed: {e}")
            return {"job_id": task.job_id, "page_num": task.page_num,
                    "pdf_path": task.pdf_path, "status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Job Manager (unchanged from rapid_paddle architecture)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JobsSnapshot:
    jobs: Dict[str, dict]
    totals: dict
    ts: float


class JobManager:
    def __init__(self):
        self._jobs = {}
        self._results = {}
        self._expected_pages = {}
        self._lock = threading.Lock()
        self._total_jobs = 0
        self._completed_jobs = 0
        self._total_pages_completed = 0
        self._processing_time_sum = 0.0
        self._snapshot = JobsSnapshot(jobs={}, totals={
            "total_jobs": 0, "completed_jobs": 0, "total_pages_completed": 0,
            "avg_throughput_pages_per_sec": 0.0}, ts=time.time())

    def create_job(self, pdf_paths, priority=5):
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id, "pdf_paths": pdf_paths, "priority": priority,
                "status": "queued", "total_pages": 0, "processed_pages": 0,
                "failed_pages": 0, "dropped_pages": 0,
                "submit_time": time.time(), "start_time": None, "end_time": None}
            self._results[job_id] = {}
            self._expected_pages[job_id] = {}
            self._total_jobs += 1
            self._rebuild_snapshot()
        return job_id

    def add_expected_pages_batch(self, job_id, pdf_path, page_nums):
        with self._lock:
            self._expected_pages.setdefault(job_id, {}).setdefault(pdf_path, set()).update(page_nums)

    def get_missing_pages(self, job_id):
        with self._lock:
            return {p: sorted(s) for p, s in self._expected_pages.get(job_id, {}).items() if s}

    def update_job(self, job_id, **kw):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kw)
                self._rebuild_snapshot()

    def _apply_one_result(self, r):
        jid, pnum, ppath = r.get("job_id"), r.get("page_num"), r.get("pdf_path")
        if not jid or ppath is None or jid not in self._results:
            return
        self._results[jid].setdefault(ppath, {})[pnum] = r
        if jid in self._expected_pages and ppath in self._expected_pages[jid]:
            self._expected_pages[jid][ppath].discard(pnum)
        if jid in self._jobs:
            j = self._jobs[jid]
            j["processed_pages"] += 1
            if r.get("status") == "error":
                j["failed_pages"] += 1
            elif r.get("status") == "success":
                self._total_pages_completed += 1
            if self._check_complete(jid):
                missing = sum(len(s) for s in self._expected_pages.get(jid, {}).values())
                if missing > 0:
                    j["status"] = "partial"
                elif j["dropped_pages"] > 0 or j["failed_pages"] > 0:
                    j["status"] = "completed_with_errors"
                else:
                    j["status"] = "completed"
                j["end_time"] = time.time()
                self._completed_jobs += 1
                if j["start_time"]:
                    self._processing_time_sum += j["end_time"] - j["start_time"]

    def apply_results_batch(self, results):
        if not results:
            return
        with self._lock:
            for r in results:
                try:
                    self._apply_one_result(r)
                except Exception as e:
                    logger.error(f"Result apply error: {e}")
            self._rebuild_snapshot()

    def _check_complete(self, jid):
        if jid not in self._expected_pages:
            return self._jobs[jid]["processed_pages"] >= self._jobs[jid]["total_pages"]
        return all(not s for s in self._expected_pages[jid].values())

    def _rebuild_snapshot(self):
        sj = {}
        for jid, j in self._jobs.items():
            sj[jid] = {k: j[k] for k in ("id", "status", "total_pages", "processed_pages",
                        "failed_pages", "dropped_pages", "priority", "submit_time", "start_time", "end_time")}
        tp = self._total_pages_completed / self._processing_time_sum if self._processing_time_sum > 0 else 0.0
        self._snapshot = JobsSnapshot(jobs=sj, totals={
            "total_jobs": self._total_jobs, "completed_jobs": self._completed_jobs,
            "total_pages_completed": self._total_pages_completed,
            "processing_time_sum": self._processing_time_sum,
            "avg_throughput_pages_per_sec": tp}, ts=time.time())

    def get_snapshot(self):
        return self._snapshot

    def get_job_status(self, jid):
        with self._lock:
            return self._jobs.get(jid, {})

    def get_job_results(self, jid):
        with self._lock:
            return self._results.get(jid, {})

    def cleanup_old_jobs(self, max_jobs=100, max_age=1800):
        with self._lock:
            now = time.time()
            done = [(jid, j["end_time"] or j["submit_time"])
                    for jid, j in self._jobs.items()
                    if j["status"] in ("completed", "completed_with_errors", "failed")]
            done.sort(key=lambda x: x[1])
            to_rm = [jid for i, (jid, t) in enumerate(done)
                     if i < len(done) - max_jobs or now - t > max_age]
            for jid in to_rm:
                self._jobs.pop(jid, None)
                self._results.pop(jid, None)
                self._expected_pages.pop(jid, None)
            if to_rm:
                self._rebuild_snapshot()
            return len(to_rm)


# ---------------------------------------------------------------------------
# Server lifecycle & endpoints
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global page_queue, result_queue, job_manager, buffer_manager

    logger.info(f"Starting PP-OCRv4 TensorRT server: {CONFIG['INSTANCE_ID']}")
    logger.info(f"ONNX Runtime {ort.__version__}, available providers: {ort.get_available_providers()}")

    # Warm up TRT engines in the main process BEFORE spawning workers.
    # This compiles engines sequentially (no GPU contention) and caches them.
    # Workers then load pre-compiled engines from cache instantly.
    warmup_trt_engines(CONFIG)

    job_manager = JobManager()
    buffer_manager = SharedBufferPool(CONFIG["BUFFER_POOL_SIZE"], BUFFER_SIZE, CONFIG["INSTANCE_ID"])

    try:
        gpus = GPUtil.getGPUs()
        available_gpus = [g.id for g in gpus if g.memoryFree > 1000] or [0]
    except:
        available_gpus = [0]
    num_gpus = len(available_gpus)

    workers_per_gpu = CONFIG["NUM_WORKERS"] / num_gpus
    actual_queue_size = min(CONFIG["PAGE_QUEUE_SIZE"],
                           int(CONFIG["BUFFER_POOL_SIZE"] * (workers_per_gpu + 1)))
    page_queue = mp.Queue(maxsize=actual_queue_size)
    result_queue = mp.Queue()

    workers = []
    for i in range(CONFIG["NUM_WORKERS"]):
        gpu_id = available_gpus[i % num_gpus]
        w = PageOCRWorker(i, page_queue, result_queue, buffer_manager, CONFIG, gpu_id)
        w.start()
        workers.append(w)
        if i < CONFIG["NUM_WORKERS"] - 1:
            time.sleep(2)  # Stagger worker startup to reduce GPU contention
    app.state.workers = workers

    threading.Thread(target=process_results, daemon=True).start()
    global dispatcher_thread
    dispatcher_thread = threading.Thread(target=dispatcher_loop, daemon=True, name="Dispatcher")
    dispatcher_thread.start()
    logger.info(f"Server started with {CONFIG['NUM_WORKERS']} workers on {num_gpus} GPU(s)")


def process_results():
    global result_queue, job_manager
    cleanup_ctr = 0
    while True:
        batch = []
        try:
            r = result_queue.get(timeout=0.1)
            if r is None:
                break
            batch.append(r)
            t0 = time.time()
            while len(batch) < 500 and time.time() - t0 < 0.05:
                try:
                    r = result_queue.get_nowait()
                    if r is None:
                        break
                    batch.append(r)
                except queue.Empty:
                    break
        except queue.Empty:
            continue
        if batch:
            job_manager.apply_results_batch(batch)
            cleanup_ctr += 1
            if cleanup_ctr >= 100:
                cleanup_ctr = 0
                job_manager.cleanup_old_jobs()


@app.on_event("shutdown")
async def shutdown_event():
    dispatcher_stop.set()
    if dispatcher_thread:
        dispatcher_thread.join(timeout=5)
    for _ in app.state.workers:
        page_queue.put(None)
    for w in app.state.workers:
        w.join(timeout=5)
        if w.is_alive():
            w.terminate()
    if buffer_manager:
        buffer_manager.cleanup()


@app.post("/process", response_model=OCRResponse)
async def process_pdfs(request: OCRRequest):
    try:
        job_id = await asyncio.to_thread(job_manager.create_job, request.pdf_paths, request.priority)
        dispatch_queue.put(EnqueueRequest(job_id, request.pdf_paths, request.priority, request.dpi))
        return OCRResponse(job_id=job_id, status="queued", message=f"Processing {len(request.pdf_paths)} PDFs")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def queue_pdf_pages(job_id, pdf_paths, priority, dpi=None):
    total_expected = queued = dropped = 0
    for pdf_path in pdf_paths:
        try:
            if not os.path.exists(pdf_path):
                continue
            pdf = pdfium.PdfDocument(pdf_path)
            n = len(pdf)
            pdf.close()
            total_expected += n
            if n > 30:
                pages = list(range(25)) + list(range(n - 5, n))
            else:
                pages = list(range(n))
            job_manager.add_expected_pages_batch(job_id, pdf_path, pages)
            for pg in pages:
                retries, buf = 0, None
                while retries < 5 and buf is None:
                    buf = buffer_manager.get_buffer(timeout=10 + retries * 10)
                    retries += 1
                if buf is None:
                    dropped += 1
                    continue
                page_queue.put(PageTask(job_id, pdf_path, pg, n, buf, (0, 0, 0), priority, time.time(), dpi))
                queued += 1
        except Exception as e:
            logger.error(f"Queue PDF {pdf_path}: {e}")
    if queued == 0:
        job_manager.update_job(job_id, status="completed", total_pages=0, processed_pages=0,
                               dropped_pages=dropped, start_time=time.time(), end_time=time.time())
    else:
        job_manager.update_job(job_id, status="processing", total_pages=queued,
                               dropped_pages=dropped, start_time=time.time())
    logger.info(f"Queued {queued}/{total_expected} pages for job {job_id}")


def dispatcher_loop():
    while not dispatcher_stop.is_set():
        try:
            req = dispatch_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Dispatcher queue error: {e}", exc_info=True)
            continue
        try:
            queue_pdf_pages(req.job_id, req.pdf_paths, req.priority, req.dpi)
        except Exception as e:
            logger.error(f"Dispatcher error job {req.job_id}: {e}", exc_info=True)
            # Mark job as failed so it doesn't stay "queued" forever
            try:
                job_manager.update_job(req.job_id, status="failed",
                                       start_time=time.time(), end_time=time.time())
            except Exception:
                pass


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    snap = job_manager.get_snapshot()
    job = snap.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    resp = dict(job)
    missing = await asyncio.to_thread(job_manager.get_missing_pages, job_id)
    if missing:
        resp["missing_pages"] = missing
        resp["missing_count"] = sum(len(v) for v in missing.values())
    return JSONResponse(content=resp)


@app.get("/results/{job_id}")
async def get_job_results(job_id: str):
    job = await asyncio.to_thread(job_manager.get_job_status, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("completed", "partial", "completed_with_errors"):
        return JSONResponse(content={"status": job["status"], "message": f"Job is {job['status']}"})
    results = await asyncio.to_thread(job_manager.get_job_results, job_id)

    def serialize(obj):
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(i) for i in obj]
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "__float__"):
            return float(obj)
        return obj

    return JSONResponse(content={"job_id": job_id, "status": "completed",
        "total_pages": job["total_pages"], "process_time": job["end_time"] - job["start_time"],
        "results": serialize(results)})


@app.get("/health")
async def health_check():
    try:
        gpus = GPUtil.getGPUs()
        gpu_stats = [{"name": g.name, "load": f"{g.load*100:.1f}%",
                      "memory": f"{g.memoryUsed}/{g.memoryTotal}MB",
                      "temperature": f"{g.temperature}°C"} for g in gpus]
    except Exception:
        gpu_stats = [{"error": "NVML unavailable"}]
    return JSONResponse(content={
        "status": "healthy", "instance_id": CONFIG["INSTANCE_ID"],
        "engine": "PP-OCRv4 + TensorRT (direct ONNX Runtime)",
        "ort_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": psutil.virtual_memory().percent,
        "gpu_stats": gpu_stats, "workers": CONFIG["NUM_WORKERS"]})


@app.get("/stats")
async def get_stats():
    snap = job_manager.get_snapshot()
    return JSONResponse(content={
        "instance_id": CONFIG["INSTANCE_ID"],
        "total_jobs": snap.totals.get("total_jobs", 0),
        "completed_jobs": snap.totals.get("completed_jobs", 0),
        "total_pages_completed": snap.totals.get("total_pages_completed", 0),
        "average_throughput": f"{snap.totals.get('avg_throughput_pages_per_sec', 0.0):.2f} pages/sec",
        "configuration": CONFIG})


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    uvicorn.run(app, host="0.0.0.0", port=CONFIG["SERVER_PORT"], log_level="info")
