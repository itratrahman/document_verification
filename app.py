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
from typing import Any, Dict, Optional  # Type hints for clearer API contracts

# Third-party web framework imports --------------------------------------------

from fastapi import FastAPI, HTTPException  # FastAPI app object and HTTP errors
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


# ------------------------------------------------------------------------------
# Helper: decode base64 image safely
# ------------------------------------------------------------------------------

def decode_base64_image(b64: str, max_bytes: int = 8 * 1024 * 1024) -> Image.Image:
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
        img = Image.open(io.BytesIO(decoded)).convert('RGB')
    except Exception:
        # Bytes were decoded, but they weren't a valid image format (png/jpg/etc.).
        raise HTTPException(status_code=400, detail="decoded data is not a valid image")

    return img  # Return PIL image to be used in downstream inference


# ------------------------------------------------------------------------------
# Stage 1: Binary classifier inference
# ------------------------------------------------------------------------------

def run_classifier(img: Image.Image, thresh: float = 0.5) -> Dict[str, Any]:
    """
    Run the binary classifier on a PIL image.

    Returns a dictionary with:
    - predicted_label: argmax class index (int)
    - probabilities: softmax probabilities (list[float] length=2)
    - passed: True if "positive" probability (index 1) >= thresh
    """
    # Retrieve shared objects prepared at startup.
    model = app.state.classifier  # The loaded PyTorch model
    transform = app.state.transform  # The preprocessing pipeline
    device = app.state.device  # "cuda" or "cpu"

    # Apply preprocessing, add batch dimension, and move to correct device.
    tensor = transform(img).unsqueeze(0).to(device)

    # Inference: no gradients needed, reduces memory use and speeds up execution.
    with torch.no_grad():
        logits = model(tensor)  # Raw unnormalized model outputs (shape: [1, 2])
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()  # Convert to Python list
        pred = int(torch.argmax(logits, dim=1).item())  # Predicted class index

    # Convention used here:
    # - class 0 = "negative / not document" (example)
    # - class 1 = "positive / document" (example)
    #
    # `passed` checks the probability of class 1 against the threshold.
    return {
        "predicted_label": pred,
        "probabilities": probs,
        "passed": (probs[1] >= thresh),
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
    ocr = app.state.ocr  # OCR engine (or None if unavailable)
    if ocr is None:
        # Service is running but OCR dependency is missing.
        return {"available": False, "message": "PaddleOCR not installed"}

    # PaddleOCR can accept a file path or an ndarray image.
    # Convert PIL RGB -> NumPy array -> BGR (OpenCV-like) ordering.
    import numpy as np  # Local import keeps global import light if OCR isn't used
    arr = np.array(img)[:, :, ::-1].copy()  # Reverse channel order: RGB -> BGR

    # Run OCR inside try/except to return a graceful error rather than crash the API.
    try:
        result = ocr.ocr(arr)  # PaddleOCR output shape varies by version/config
        # This code assumes a dict-like format for recognized texts.
        # If your PaddleOCR returns list-of-lines (common), you may need to adapt this.
        texts = result[0]["rec_texts"] if result and isinstance(result, list) and len(result) > 0 else []
    except Exception as e:
        # OCR engine is available but failed during processing.
        return {"available": True, "error": str(e)}

    # Rule-based marker detection: build a single normalized string for regex search.
    import re  # Regex utilities for robust pattern matching
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

    # Return structured outputs to help debugging and client-side UI.
    return {
        "available": True,  # OCR engine exists and ran (even if no text found)
        "raw_text": texts,  # Raw recognized lines (as provided by PaddleOCR parsing above)
        "found_markers": [m for m in markers if found.get(m, False)],  # Markers detected
        "missing_markers": missing,  # Markers absent
        "is_valid_format": len(missing) == 0,  # True only if all markers found
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
    RF = app.state.retinaface  # Stored RetinaFace reference (or None if unavailable)
    if RF is None:
        # Service is running but face detector dependency is missing.
        return {"available": False, "message": "RetinaFace not installed"}

    # Convert PIL image to NumPy RGB array for PyTorch model
    import numpy as np  # Local import to keep startup lighter
    import torch

    arr = np.array(img)  # Keep as RGB for retinaface-pytorch

    # Detect faces using PyTorch RetinaFace
    try:
        with torch.no_grad():
            detections_raw = RF.predict_jsons(arr, confidence_threshold=0.5)
    except Exception as e:
        return {"available": False, "message": f"Face detection failed: {str(e)}"}

    # Normalize detections into a list of faces with a score + bounding box.
    faces = []
    for det in detections_raw:
        score = float(det.get("score", 0.0))  # Confidence score
        bbox = det.get("bbox", [])  # Bounding box [x1, y1, x2, y2]
        if len(bbox) == 4:
            faces.append({
                "score": score,
                "box": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            })

    # Basic verification: compute relative face size compared to total image area.
    h, w = arr.shape[:2]  # Image height/width
    img_area = float(h * w)  # Total pixel area (as float for division safety)

    # Case 1: no face found.
    if len(faces) == 0:
        return {
            "available": True,
            "ok": False,
            "reason": "no_faces",
            "num_faces": 0,
            "faces": faces,
        }

    # Case 2: multiple faces found.
    if len(faces) > 1:
        return {
            "available": True,
            "ok": False,
            "reason": "multiple_faces",
            "num_faces": len(faces),
            "faces": faces,
        }

    # Exactly one face found: compute bounding box area.
    box = faces[0]["box"]  # [x1, y1, x2, y2]
    fa = max(0, box[2] - box[0]) * max(0, box[3] - box[1])  # Face box area in pixels

    # Relative area of face compared to entire image.
    rel_area = fa / img_area if img_area > 0 else 0.0

    # Reject extremely small faces (likely false positive or far-away person).
    if rel_area < 0.02:
        return {
            "available": True,
            "ok": False,
            "reason": "face_too_small",
            "num_faces": 1,
            "faces": faces,
        }

    # Reject extremely large faces (likely over-cropped or incorrect detection).
    if rel_area > 0.6:
        return {
            "available": True,
            "ok": False,
            "reason": "face_too_large",
            "num_faces": 1,
            "faces": faces,
        }

    # Passed all face checks.
    return {
        "available": True,
        "ok": True,
        "reason": "single_face_ok",
        "num_faces": 1,
        "faces": faces,
    }


# ------------------------------------------------------------------------------
# API endpoint: /verify
# ------------------------------------------------------------------------------

@app.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest):
    """
    Main endpoint: verifies an incoming base64 image.

    Implementation notes:
    - The handler is `async`, but ML inference is CPU/GPU-bound and blocking.
      Therefore we use `run_in_threadpool(...)` to keep the event loop responsive.
    - All three checks are executed for transparency, even if the classifier fails,
      so the client can see what happened (OCR/face outputs can be helpful).
    """
    # Decode base64 -> PIL image.
    # This is blocking work (base64 decode, PIL decode), so run it in the threadpool.
    img = await run_in_threadpool(decode_base64_image, req.image_base64)

    # Stage 1: Binary classifier (blocking torch inference -> threadpool).
    binary = await run_in_threadpool(run_classifier, img, req.thresh_binary)

    # Stage 2: OCR + markers (blocking -> threadpool).
    ocr = await run_in_threadpool(run_ocr_verification, img)

    # Stage 3: Face detection (blocking -> threadpool).
    face = await run_in_threadpool(run_face_verification, img)

    # Overall decision:
    # - classifier must pass threshold
    # - OCR must find all markers
    # - face check must pass
    ok = bool(
        binary.get("passed", False)
        and ocr.get("is_valid_format", False)
        and face.get("ok", False)
    )

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
