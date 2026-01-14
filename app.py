"""
app.py — FastAPI AI inference server (document verification)

This service exposes a single POST endpoint: /verify

Given an input image (base64-encoded), the server runs three independent checks:

1) Binary image classifier (e.g., EfficientNet-B0) to decide whether the image looks like
   the expected document type.
2) OCR (PaddleOCR) to extract text and verify presence of required "marker" fields
   (e.g., "1", "2", "3", "4a", ...).
3) Face detection (RetinaFace) to ensure a single reasonably-sized face is present.

The result combines these checks into a single boolean `ok`.

Notes for readers:
- Heavy ML dependencies (PaddleOCR / RetinaFace) are imported lazily at startup so the
  API can still run even if they are not installed.
- CPU-bound work (PIL decoding, Torch inference, OCR, face detection) is executed in a
  threadpool to avoid blocking the FastAPI async event loop.

This file is intentionally "heavily commented" to serve as living documentation.
"""

# Standard library imports ------------------------------------------------------

import os  # File-path utilities (checkpoint discovery, temp file cleanup, etc.)
import io  # In-memory byte buffers used to decode base64 images
import base64  # Base64 decoding for image transport over JSON
import logging  # Structured logging via Uvicorn's logger
import hashlib  # SHA256 hashing for image deduplication
import uuid  # Unique request ID generation
import time  # Performance timing
import socket  # Hostname detection
from datetime import datetime  # Timestamp generation
from typing import Any, Dict, Optional, Tuple  # Type hints for clearer API contracts

# Third-party web framework imports --------------------------------------------

from fastapi import FastAPI, HTTPException, Request  # FastAPI app object and HTTP errors
from pydantic import BaseModel  # Request/response validation + OpenAPI schemas
from starlette.concurrency import run_in_threadpool  # Run blocking code in threads

# ML / vision imports -----------------------------------------------------------

import torch  # PyTorch inference and device management
from torchvision import transforms  # Standard image transforms (resize/crop/normalize)
from PIL import Image  # PIL image object for decoding and preprocessing

# Local imports -----------------------------------------------------------------
# `create_model` is expected to build your binary classifier architecture (e.g., EfficientNet).
# Keeping this in a separate module makes the server small and reusable.
from model import create_model

# ------------------------------------------------------------------------------
# App + logging setup
# ------------------------------------------------------------------------------

# Use Uvicorn's logger name so logs appear consistently when you run:
#   uvicorn app:app --reload
logger = logging.getLogger("uvicorn")

# Create the FastAPI application. `title` shows up in Swagger UI (/docs).
app = FastAPI(title="Document Verification API")

# API version for tracking schema changes over time
API_VERSION = "1.0.0"

# MongoDB imports (optional dependency)
try:
    from motor.motor_asyncio import AsyncIOMotorClient  # Async MongoDB driver
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger.warning("motor not installed - MongoDB logging will be disabled")


# ------------------------------------------------------------------------------
# Request/Response schemas
# ------------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    """
    Request schema for /verify.

    - image_base64: image payload as base64 string (optionally prefixed with a data-URI header)
      Example:
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    - thresh_binary: threshold used to decide whether classifier "passes".
    """
    image_base64: str  # The image as base64 text (required)
    thresh_binary: Optional[float] = 0.5  # Default acceptance threshold for classifier


class VerifyResponse(BaseModel):
    """
    Response schema returned by /verify.

    - ok: overall decision (true only if all checks pass)
    - binary: details from the classifier stage
    - ocr: details from OCR + marker verification stage
    - face: details from face-detection stage
    """
    ok: bool  # Final decision
    binary: Dict[str, Any]  # Classifier outputs (probabilities, predicted label, etc.)
    ocr: Dict[str, Any]  # OCR outputs (raw text, found markers, etc.)
    face: Dict[str, Any]  # Face-detection outputs (number of faces, boxes, etc.)


# ------------------------------------------------------------------------------
# Startup hook: load models once and store them on app.state
# ------------------------------------------------------------------------------

@app.on_event("startup")
def load_models():
    """
    Load all AI models once on process startup.

    Why do this at startup?
    - Loading weights and initializing OCR/face models can be slow.
    - Doing it once improves request latency and avoids repeated GPU/CPU allocations.

    Everything is stored into `app.state` so all request handlers can access it.
    """
    # Decide compute device: prefer GPU (CUDA) if available; otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device  # Save device for later use in inference
    logger.info(f"Using device: {device}")  # Log the chosen device for debugging

    # ----------------------------
    # Binary classifier (EfficientNet-B0 or similar)
    # ----------------------------
    try:
        # Build the model architecture (must match the checkpoint weights).
        model = create_model(num_classes=2)

        # Potential checkpoint locations (first existing file is used).
        ckpt_paths = [
            os.path.join("models", "best_efficientnet_binary.pt"),
            os.path.join("models", "final_efficientnet_binary.pt"),
        ]

        # Attempt to load weights from disk if any known checkpoint exists.
        for p in ckpt_paths:
            if os.path.exists(p):  # Checkpoint exists on filesystem?
                state = torch.load(p, map_location=device)  # Load to CPU/GPU safely
                model.load_state_dict(state)  # Restore model weights
                logger.info(f"Loaded classifier weights from {p}")  # Confirm load
                break  # Stop after first successful load

        # Put model in inference mode (disables dropout, uses running stats for BN, etc.).
        model.eval()

        # Move model to the selected device (GPU if available).
        model.to(device)

        # Store on app.state so request handlers can access it.
        app.state.classifier = model
    except Exception:
        # If model init fails, the API should not start (better than serving broken inference).
        logger.exception("Failed to initialize classifier")
        raise  # Re-raise so FastAPI/Uvicorn fails fast and shows the stack trace

    # ----------------------------
    # Common image transform for classifier inference
    # ----------------------------
    # The following are typical ImageNet transforms:
    # - Resize shorter side to 256
    # - Center crop to 224x224
    # - Convert to tensor and normalize to ImageNet mean/std
    #
    # This *must* match how the classifier was trained.
    app.state.transform = transforms.Compose([
        transforms.Resize(256),  # Resize so center crop has enough pixels
        transforms.CenterCrop(224),  # Crop to model input resolution
        transforms.ToTensor(),  # Convert PIL image (0-255) -> torch tensor (0-1)
        transforms.Normalize(  # Normalize channels like ImageNet pretraining
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ----------------------------
    # PaddleOCR initialization (optional dependency)
    # ----------------------------
    try:
        # Lazy import: avoids import-time failures if PaddleOCR isn't installed.
        from paddleocr import PaddleOCR

        # Initialize OCR engine.
        # use_angle_cls=True helps with rotated/scanned documents.
        # lang='en' selects English OCR model.
        app.state.ocr = PaddleOCR(use_angle_cls=True, lang='en')

        logger.info("PaddleOCR initialized")
    except Exception as e:
        # If OCR isn't installed, we keep the service alive and return "available=False" later.
        logger.warning(f"PaddleOCR not available: {e}")
        app.state.ocr = None  # Signal to downstream functions that OCR is disabled

    # ----------------------------
    # RetinaFace initialization (optional dependency)
    # ----------------------------
    try:
        # Lazy import: avoids import-time failures if retinaface isn't installed.
        from retinaface.pre_trained_models import get_model
        
        # Initialize RetinaFace model with PyTorch backend
        app.state.retinaface = get_model("resnet50_2020-07-20", max_size=2048)
        app.state.retinaface.eval()

        logger.info("RetinaFace available")
    except Exception as e:
        # If face detection isn't installed, we keep the service alive and return "available=False".
        logger.warning(f"RetinaFace not available: {e}")
        app.state.retinaface = None  # Signal to downstream functions that face check is disabled

    # ----------------------------
    # MongoDB connection (optional)
    # ----------------------------
    if MONGODB_AVAILABLE:
        try:
            # Read MongoDB connection details from environment variables
            mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
            mongodb_db = os.getenv("MONGODB_DATABASE", "document_verification")
            
            # Create async MongoDB client
            app.state.mongo_client = AsyncIOMotorClient(mongodb_uri)
            app.state.mongo_db = app.state.mongo_client[mongodb_db]
            app.state.mongo_collection = app.state.mongo_db["inference_logs"]
            
            # Test connection with a simple ping
            app.state.mongo_client.admin.command('ping')
            logger.info(f"MongoDB connected: {mongodb_db}")
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}")
            app.state.mongo_client = None
            app.state.mongo_db = None
            app.state.mongo_collection = None
    else:
        app.state.mongo_client = None
        app.state.mongo_db = None
        app.state.mongo_collection = None

    # Store environment information for logging
    app.state.hostname = socket.gethostname()
    app.state.container_id = os.getenv("HOSTNAME", None)  # Docker container ID
    if torch.cuda.is_available():
        app.state.gpu_name = torch.cuda.get_device_name(0)
    else:
        app.state.gpu_name = None


# ------------------------------------------------------------------------------
# Helper: compute SHA256 hash of image bytes
# ------------------------------------------------------------------------------

def compute_image_hash(image_bytes: bytes) -> str:
    """Compute SHA256 hash of image bytes for deduplication."""
    return hashlib.sha256(image_bytes).hexdigest()


# ------------------------------------------------------------------------------
# Helper: decode base64 image safely (enhanced with metadata)
# ------------------------------------------------------------------------------

def decode_base64_image(b64: str, max_bytes: int = 8 * 1024 * 1024) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Decode a base64-encoded image and return a PIL Image.

    Security / robustness features:
    - Accepts "data:image/...;base64,AAAA" headers; strips them if present.
    - Enforces a maximum decoded payload size (`max_bytes`) to prevent memory abuse.
    - Raises HTTPException(400) for invalid base64 or invalid image bytes.
    """
    try:
        # Some clients send a data-URI prefix like "data:image/png;base64,...".
        # This finds the comma and strips everything before it.
        header_sep = b64.find(',')
        if header_sep != -1:  # If comma is present, assume data URI format
            b64 = b64[header_sep + 1:]  # Keep only the base64 payload

        # Decode base64 text -> raw bytes.
        decoded = base64.b64decode(b64)
    except Exception:
        # Any base64 error results in a "Bad Request" for client clarity.
        raise HTTPException(status_code=400, detail="invalid base64 image")

    # Enforce size limit to prevent extremely large requests from exhausting memory.
    if len(decoded) > max_bytes:
        raise HTTPException(status_code=400, detail="image exceeds maximum allowed size")

    try:
        # Open bytes as an image and standardize into RGB (3 channels).
        img_stream = io.BytesIO(decoded)
        img_pil = Image.open(img_stream)
        img_format = img_pil.format  # Get original format before conversion
        img = img_pil.convert('RGB')
    except Exception:
        # Bytes were decoded, but they weren't a valid image format (png/jpg/etc.).
        raise HTTPException(status_code=400, detail="decoded data is not a valid image")

    # Compute metadata for logging
    metadata = {
        "image_hash": compute_image_hash(decoded),
        "image_size_bytes": len(decoded),
        "image_dimensions": {"width": img.width, "height": img.height},
        "image_format": img_format if img_format else "UNKNOWN"
    }

    return img, metadata  # Return PIL image and metadata


# ------------------------------------------------------------------------------
# MongoDB logging helper
# ------------------------------------------------------------------------------

async def log_inference_to_mongodb(log_document: Dict[str, Any]):
    """Async function to log inference data to MongoDB."""
    if app.state.mongo_collection is None:
        return  # MongoDB not available, skip logging
    
    try:
        await app.state.mongo_collection.insert_one(log_document)
        logger.debug(f"Logged inference request {log_document['request_id']} to MongoDB")
    except Exception as e:
        logger.error(f"Failed to log to MongoDB: {e}")


# ------------------------------------------------------------------------------
# Stage 1: Binary classifier inference (enhanced with timing)
# ------------------------------------------------------------------------------

def run_classifier(img: Image.Image, thresh: float = 0.5) -> Dict[str, Any]:
    """
    Run the binary classifier on a PIL image.

    Returns a dictionary with:
    - predicted_label: argmax class index (int)
    - probabilities: softmax probabilities (list[float] length=2)
    - passed: True if "positive" probability (index 1) >= thresh
    - timing and performance metrics
    """
    stage_start = time.time()
    
    # Retrieve shared objects prepared at startup.
    model = app.state.classifier  # The loaded PyTorch model
    transform = app.state.transform  # The preprocessing pipeline
    device = app.state.device  # "cuda" or "cpu"

    # Preprocessing timing
    preprocess_start = time.time()
    tensor = transform(img).unsqueeze(0).to(device)
    preprocess_time = (time.time() - preprocess_start) * 1000

    # Inference timing
    inference_start = time.time()
    with torch.no_grad():
        logits = model(tensor)  # Raw unnormalized model outputs (shape: [1, 2])
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()  # Convert to Python list
        pred = int(torch.argmax(logits, dim=1).item())  # Predicted class index
    inference_time = (time.time() - inference_start) * 1000

    # Postprocessing timing (minimal in this case)
    postprocess_start = time.time()
    passed = probs[1] >= thresh
    confidence = max(probs)
    postprocess_time = (time.time() - postprocess_start) * 1000

    total_time = (time.time() - stage_start) * 1000

    # Memory tracking (if CUDA is available)
    memory_allocated_mb = None
    if torch.cuda.is_available():
        memory_allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)

    return {
        "predicted_label": pred,
        "probabilities": probs,
        "confidence": confidence,
        "passed": passed,
        "threshold_used": thresh,
        "duration_ms": total_time,
        "performance": {
            "preprocessing_ms": preprocess_time,
            "inference_ms": inference_time,
            "postprocessing_ms": postprocess_time,
            "memory_allocated_mb": memory_allocated_mb
        }
    }


# ------------------------------------------------------------------------------
# Stage 2: OCR + simple rule-based verification
# ------------------------------------------------------------------------------

def run_ocr_verification(img: Image.Image) -> Dict[str, Any]:
    """
    Run OCR and verify document format by searching for expected markers.

    This is intentionally a simple heuristic check:
    - Run OCR to collect recognized text.
    - Normalize and search for markers like "1", "2", "3", "4a"... that are commonly
      printed near fields on certain IDs/licenses.
    - If all markers are found, return `is_valid_format=True`.

    If PaddleOCR isn't installed or fails to initialize, returns available=False.
    """
    stage_start = time.time()
    
    ocr = app.state.ocr  # OCR engine (or None if unavailable)
    if ocr is None:
        # Service is running but OCR dependency is missing.
        return {
            "available": False,
            "message": "PaddleOCR not installed",
            "duration_ms": 0,
            "status": "skipped"
        }

    # PaddleOCR can accept a file path or an ndarray image.
    # Convert PIL RGB -> NumPy array -> BGR (OpenCV-like) ordering.
    import numpy as np  # Local import keeps global import light if OCR isn't used
    import re  # Regex utilities for robust pattern matching
    
    preprocess_start = time.time()
    arr = np.array(img)[:, :, ::-1].copy()  # Reverse channel order: RGB -> BGR
    preprocess_time = (time.time() - preprocess_start) * 1000

    # Run OCR inside try/except to return a graceful error rather than crash the API.
    ocr_start = time.time()
    try:
        result = ocr.ocr(arr)  # PaddleOCR output shape varies by version/config
        # This code assumes a dict-like format for recognized texts.
        # If your PaddleOCR returns list-of-lines (common), you may need to adapt this.
        texts = result[0]["rec_texts"] if result and isinstance(result, list) and len(result) > 0 else []
    except Exception as e:
        # OCR engine is available but failed during processing.
        return {
            "available": True,
            "error": str(e),
            "duration_ms": (time.time() - stage_start) * 1000,
            "status": "failed"
        }
    ocr_time = (time.time() - ocr_start) * 1000

    # Rule-based marker detection: build a single normalized string for regex search.
    text_processing_start = time.time()
    text = " ".join(texts or [])  # Join all lines into one string
    text = re.sub(r"\s+", " ", text).strip().lower()  # Normalize whitespace + lowercase

    # Markers to detect (intentionally a small set, can be expanded as needed).
    markers = ["1", "2", "3", "4a", "4b", "4c", "4d", "5", "7", "8", "9"]

    found: Dict[str, bool] = {}  # Track which markers are detected
    for m in markers:
        # For single-digit markers, avoid matching parts of larger numbers.
        if m.isdigit():
            # Example pattern for marker "1": match standalone "1" not preceded by digit,
            # followed by typical punctuation like ".", ")", ":" etc.
            pattern = rf"(?<!\d){m}\s*[\.\):,\-]"
        else:
            # For alphanumeric markers like "4a", use word boundaries.
            pattern = rf"\b{re.escape(m)}\b\s*[\.\):,\-]?"
        # Perform case-insensitive search (even though we lowercased, this is extra safety).
        found[m] = re.search(pattern, text, flags=re.IGNORECASE) is not None

    # Markers not found are considered "missing".
    missing = [m for m in markers if not found.get(m, False)]
    text_processing_time = (time.time() - text_processing_start) * 1000

    total_time = (time.time() - stage_start) * 1000

    # Return structured outputs to help debugging and client-side UI.
    return {
        "available": True,  # OCR engine exists and ran (even if no text found)
        "status": "success",
        "raw_text": texts,  # Raw recognized lines (as provided by PaddleOCR parsing above)
        "text_blocks_count": len(texts),
        "total_characters": sum(len(t) for t in texts),
        "found_markers": [m for m in markers if found.get(m, False)],  # Markers detected
        "missing_markers": missing,  # Markers absent
        "found_count": len([m for m in markers if found.get(m, False)]),
        "expected_count": len(markers),
        "is_valid_format": len(missing) == 0,  # True only if all markers found
        "normalized_text_sample": text[:200],  # First 200 characters
        "duration_ms": total_time,
        "performance": {
            "image_preprocessing_ms": preprocess_time,
            "ocr_inference_ms": ocr_time,
            "text_processing_ms": text_processing_time
        }
    }


# ------------------------------------------------------------------------------
# Stage 3: Face verification using RetinaFace (face detection)
# ------------------------------------------------------------------------------

def run_face_verification(img: Image.Image) -> Dict[str, Any]:
    """
    Run face detection and validate a simple policy:
    - Exactly 1 face detected
    - Face bounding box area is within a reasonable fraction of the image:
        - not too small (likely noise)
        - not too large (likely incorrect detection / cropped face)

    If RetinaFace isn't installed or fails to initialize, returns available=False.
    """
    stage_start = time.time()
    
    RF = app.state.retinaface  # Stored RetinaFace reference (or None if unavailable)
    if RF is None:
        # Service is running but face detector dependency is missing.
        return {
            "available": False,
            "message": "RetinaFace not installed",
            "duration_ms": 0,
            "status": "skipped"
        }

    # Convert PIL image to NumPy RGB array for PyTorch model
    import numpy as np  # Local import to keep startup lighter
    
    preprocess_start = time.time()
    arr = np.array(img)  # Keep as RGB for retinaface-pytorch
    h, w = arr.shape[:2]  # Image height/width
    img_area = float(h * w)  # Total pixel area (as float for division safety)
    preprocess_time = (time.time() - preprocess_start) * 1000

    # Detect faces using PyTorch RetinaFace
    detection_start = time.time()
    try:
        with torch.no_grad():
            detections_raw = RF.predict_jsons(arr, confidence_threshold=0.5)
    except Exception as e:
        return {
            "available": False,
            "message": f"Face detection failed: {str(e)}",
            "duration_ms": (time.time() - stage_start) * 1000,
            "status": "failed"
        }
    detection_time = (time.time() - detection_start) * 1000

    # Normalize detections into a list of faces with a score + bounding box.
    postprocess_start = time.time()
    faces = []
    for det in detections_raw:
        score = float(det.get("score", 0.0))  # Confidence score
        bbox = det.get("bbox", [])  # Bounding box [x1, y1, x2, y2]
        if len(bbox) == 4:
            box_coords = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            face_area = max(0, box_coords[2] - box_coords[0]) * max(0, box_coords[3] - box_coords[1])
            rel_area = face_area / img_area if img_area > 0 else 0.0
            aspect_ratio = (box_coords[3] - box_coords[1]) / max(1, box_coords[2] - box_coords[0])
            
            faces.append({
                "score": score,
                "box": box_coords,
                "area_pixels": face_area,
                "relative_area": round(rel_area, 4),
                "aspect_ratio": round(aspect_ratio, 2)
            })

    # Define validation thresholds
    MIN_RELATIVE_AREA = 0.02
    MAX_RELATIVE_AREA = 0.6
    postprocess_time = (time.time() - postprocess_start) * 1000
    total_time = (time.time() - stage_start) * 1000

    # Case 1: no face found.
    if len(faces) == 0:
        return {
            "available": True,
            "status": "success",
            "ok": False,
            "reason": "no_faces",
            "num_faces": 0,
            "faces": faces,
            "validation": {
                "face_count_valid": False,
                "size_valid": False,
                "min_relative_area": MIN_RELATIVE_AREA,
                "max_relative_area": MAX_RELATIVE_AREA
            },
            "duration_ms": total_time,
            "performance": {
                "preprocessing_ms": preprocess_time,
                "detection_ms": detection_time,
                "postprocessing_ms": postprocess_time
            }
        }

    # Case 2: multiple faces found.
    if len(faces) > 1:
        return {
            "available": True,
            "status": "success",
            "ok": False,
            "reason": "multiple_faces",
            "num_faces": len(faces),
            "faces": faces,
            "validation": {
                "face_count_valid": False,
                "size_valid": True,
                "min_relative_area": MIN_RELATIVE_AREA,
                "max_relative_area": MAX_RELATIVE_AREA
            },
            "duration_ms": total_time,
            "performance": {
                "preprocessing_ms": preprocess_time,
                "detection_ms": detection_time,
                "postprocessing_ms": postprocess_time
            }
        }

    # Exactly one face found: validate size
    rel_area = faces[0]["relative_area"]
    size_valid = MIN_RELATIVE_AREA <= rel_area <= MAX_RELATIVE_AREA

    if not size_valid:
        if rel_area < MIN_RELATIVE_AREA:
            reason = "face_too_small"
        else:
            reason = "face_too_large"
        
        return {
            "available": True,
            "status": "success",
            "ok": False,
            "reason": reason,
            "num_faces": 1,
            "faces": faces,
            "validation": {
                "face_count_valid": True,
                "size_valid": False,
                "min_relative_area": MIN_RELATIVE_AREA,
                "max_relative_area": MAX_RELATIVE_AREA,
                "actual_relative_area": rel_area
            },
            "duration_ms": total_time,
            "performance": {
                "preprocessing_ms": preprocess_time,
                "detection_ms": detection_time,
                "postprocessing_ms": postprocess_time
            }
        }

    # Passed all face checks.
    return {
        "available": True,
        "status": "success",
        "ok": True,
        "reason": "single_face_ok",
        "num_faces": 1,
        "faces": faces,
        "validation": {
            "face_count_valid": True,
            "size_valid": True,
            "min_relative_area": MIN_RELATIVE_AREA,
            "max_relative_area": MAX_RELATIVE_AREA,
            "actual_relative_area": rel_area
        },
        "duration_ms": total_time,
        "performance": {
            "preprocessing_ms": preprocess_time,
            "detection_ms": detection_time,
            "postprocessing_ms": postprocess_time
        }
    }


# ------------------------------------------------------------------------------
# API endpoint: /verify
# ------------------------------------------------------------------------------

@app.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest, request: Request):
    """
    Main endpoint: verifies an incoming base64 image.

    Implementation notes:
    - The handler is `async`, but ML inference is CPU/GPU-bound and blocking.
      Therefore we use `run_in_threadpool(...)` to keep the event loop responsive.
    - All three checks are executed for transparency, even if the classifier fails,
      so the client can see what happened (OCR/face outputs can be helpful).
    - Comprehensive logging to MongoDB tracks all stages and performance metrics.
    """
    # Generate unique request ID and track overall timing
    request_id = str(uuid.uuid4())
    request_start = time.time()
    request_timestamp = datetime.utcnow()
    
    # Extract client information
    client_info = {
        "ip_address": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent", None)
    }
    
    # Decode base64 -> PIL image.
    # This is blocking work (base64 decode, PIL decode), so run it in the threadpool.
    preprocess_start = time.time()
    img, img_metadata = await run_in_threadpool(decode_base64_image, req.image_base64)
    preprocess_time = (time.time() - preprocess_start) * 1000

    # Add image metadata to input section
    img_metadata["threshold_binary"] = req.thresh_binary
    
    # Stage 1: Binary classifier (blocking torch inference -> threadpool).
    binary_start = time.time()
    binary = await run_in_threadpool(run_classifier, img, req.thresh_binary)
    binary["started_at"] = datetime.utcfromtimestamp(binary_start)
    binary["completed_at"] = datetime.utcnow()

    # Stage 2: OCR + markers (blocking -> threadpool).
    ocr_start = time.time()
    ocr = await run_in_threadpool(run_ocr_verification, img)
    ocr["started_at"] = datetime.utcfromtimestamp(ocr_start)
    ocr["completed_at"] = datetime.utcnow()

    # Stage 3: Face detection (blocking -> threadpool).
    face_start = time.time()
    face = await run_in_threadpool(run_face_verification, img)
    face["started_at"] = datetime.utcfromtimestamp(face_start)
    face["completed_at"] = datetime.utcnow()

    # Overall decision:
    # - classifier must pass threshold
    # - OCR must find all markers
    # - face check must pass
    ok = bool(
        binary.get("passed", False)
        and ocr.get("is_valid_format", False)
        and face.get("ok", False)
    )
    
    # Calculate total request time
    total_duration = (time.time() - request_start) * 1000
    
    # Build comprehensive MongoDB log document
    log_document = {
        "request_id": request_id,
        "timestamp": request_timestamp,
        "api_version": API_VERSION,
        
        "input": img_metadata,
        
        "environment": {
            "device": str(app.state.device),
            "hostname": app.state.hostname,
            "container_id": app.state.container_id,
            "gpu_name": app.state.gpu_name,
            "model_versions": {
                "classifier": "best_efficientnet_binary.pt",
                "ocr": "paddleocr-en" if app.state.ocr else None,
                "face_detector": "resnet50_2020-07-20" if app.state.retinaface else None
            }
        },
        
        "preprocessing": {
            "started_at": datetime.utcfromtimestamp(preprocess_start),
            "completed_at": datetime.utcfromtimestamp(preprocess_start + preprocess_time/1000),
            "duration_ms": preprocess_time,
            "status": "success",
            "errors": None,
            "operations": ["base64_decode", "image_decode", "rgb_conversion"]
        },
        
        "binary_classifier": {
            "started_at": binary.get("started_at"),
            "completed_at": binary.get("completed_at"),
            "duration_ms": binary.get("duration_ms"),
            "status": "success",
            "model_name": "EfficientNet-B0",
            "checkpoint": "best_efficientnet_binary.pt",
            "preprocessing": {
                "resize_to": 256,
                "center_crop": 224,
                "normalization": "imagenet"
            },
            "predictions": {
                "predicted_label": binary.get("predicted_label"),
                "probabilities": binary.get("probabilities"),
                "confidence": binary.get("confidence"),
                "passed": binary.get("passed"),
                "threshold_used": binary.get("threshold_used")
            },
            "performance": binary.get("performance"),
            "errors": None
        },
        
        "ocr_verification": {
            "started_at": ocr.get("started_at"),
            "completed_at": ocr.get("completed_at"),
            "duration_ms": ocr.get("duration_ms"),
            "status": ocr.get("status", "skipped"),
            "available": ocr.get("available"),
            "engine": "PaddleOCR" if ocr.get("available") else None,
            "engine_version": "2.7.0" if ocr.get("available") else None,
            "config": {
                "use_angle_cls": True,
                "lang": "en"
            } if ocr.get("available") else None,
            "ocr_results": {
                "raw_text": ocr.get("raw_text"),
                "text_blocks_count": ocr.get("text_blocks_count"),
                "total_characters": ocr.get("total_characters"),
                "processing_language": "en"
            } if ocr.get("available") else None,
            "marker_validation": {
                "expected_markers": ["1", "2", "3", "4a", "4b", "4c", "4d", "5", "7", "8", "9"],
                "found_markers": ocr.get("found_markers"),
                "missing_markers": ocr.get("missing_markers"),
                "found_count": ocr.get("found_count"),
                "expected_count": ocr.get("expected_count"),
                "is_valid_format": ocr.get("is_valid_format"),
                "normalized_text_sample": ocr.get("normalized_text_sample")
            } if ocr.get("available") else None,
            "performance": ocr.get("performance"),
            "errors": ocr.get("error") if "error" in ocr else None
        },
        
        "face_detection": {
            "started_at": face.get("started_at"),
            "completed_at": face.get("completed_at"),
            "duration_ms": face.get("duration_ms"),
            "status": face.get("status", "skipped"),
            "available": face.get("available"),
            "model": "RetinaFace" if face.get("available") else None,
            "model_version": "resnet50_2020-07-20" if face.get("available") else None,
            "config": {
                "confidence_threshold": 0.5,
                "max_size": 2048
            } if face.get("available") else None,
            "detection_results": {
                "num_faces": face.get("num_faces"),
                "ok": face.get("ok"),
                "reason": face.get("reason"),
                "faces": face.get("faces")
            } if face.get("available") else None,
            "validation": face.get("validation"),
            "performance": face.get("performance"),
            "errors": face.get("message") if "message" in face and not face.get("available") else None
        },
        
        "response": {
            "timestamp": datetime.utcnow(),
            "ok": ok,
            "decision_factors": {
                "binary_passed": binary.get("passed", False),
                "ocr_passed": ocr.get("is_valid_format", False),
                "face_passed": face.get("ok", False)
            },
            "http_status": 200
        },
        
        "performance": {
            "total_duration_ms": total_duration,
            "preprocessing_ms": preprocess_time,
            "binary_classifier_ms": binary.get("duration_ms", 0),
            "ocr_verification_ms": ocr.get("duration_ms", 0),
            "face_detection_ms": face.get("duration_ms", 0),
            "breakdown": {
                "preprocessing_pct": round((preprocess_time / total_duration * 100), 1) if total_duration > 0 else 0,
                "binary_classifier_pct": round((binary.get("duration_ms", 0) / total_duration * 100), 1) if total_duration > 0 else 0,
                "ocr_verification_pct": round((ocr.get("duration_ms", 0) / total_duration * 100), 1) if total_duration > 0 else 0,
                "face_detection_pct": round((face.get("duration_ms", 0) / total_duration * 100), 1) if total_duration > 0 else 0
            }
        },
        
        "has_errors": False,
        "error_stages": [],
        
        "tags": ["production", "document_verification"],
        "client_info": client_info
    }
    
    # Log to MongoDB asynchronously (non-blocking)
    if app.state.mongo_collection is not None:
        await log_inference_to_mongodb(log_document)

    # Return a typed response that matches VerifyResponse schema.
    return VerifyResponse(ok=ok, binary=binary, ocr=ocr, face=face)


# ------------------------------------------------------------------------------
# Local development entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Running as `python app.py` will start Uvicorn programmatically.
    # In production you typically run:
    #   uvicorn app:app --host 0.0.0.0 --port 8000
    import uvicorn  # ASGI server used to run FastAPI apps

    # Start the server on all interfaces (0.0.0.0) so it is accessible externally.
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
