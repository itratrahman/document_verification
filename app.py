import os
import io
import base64
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import torch
from torchvision import transforms
from PIL import Image

# Import model creation helper from existing script
from model import create_model

# PaddleOCR and RetinaFace are optional heavy deps; import lazily in startup

logger = logging.getLogger("uvicorn")
app = FastAPI(title="Document Verification API")


class VerifyRequest(BaseModel):
    image_base64: str
    thresh_binary: Optional[float] = 0.5


class VerifyResponse(BaseModel):
    ok: bool
    binary: Dict[str, Any]
    ocr: Dict[str, Any]
    face: Dict[str, Any]


@app.on_event("startup")
def load_models():
    """Load all AI models once on startup and store on the app object."""
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    logger.info(f"Using device: {device}")

    # Binary classifier (EfficientNet-B0)
    try:
        model = create_model(num_classes=2)
        # Try loading a checkpoint if present
        ckpt_paths = [
            os.path.join("models", "best_efficientnet_binary.pt"),
            os.path.join("models", "final_efficientnet_binary.pt"),
        ]
        for p in ckpt_paths:
            if os.path.exists(p):
                state = torch.load(p, map_location=device)
                model.load_state_dict(state)
                logger.info(f"Loaded classifier weights from {p}")
                break
        model.eval()
        model.to(device)
        app.state.classifier = model
    except Exception as e:
        logger.exception("Failed to initialize classifier")
        raise

    # Common image transform for inference (center-crop style)
    app.state.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # PaddleOCR
    try:
        from paddleocr import PaddleOCR
        app.state.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        logger.info("PaddleOCR initialized")
    except Exception as e:
        logger.warning(f"PaddleOCR not available: {e}")
        app.state.ocr = None

    # RetinaFace (face detection)
    try:
        from retinaface import RetinaFace
        app.state.retinaface = RetinaFace
        logger.info("RetinaFace available")
    except Exception as e:
        logger.warning(f"RetinaFace not available: {e}")
        app.state.retinaface = None


def decode_base64_image(b64: str, max_bytes: int = 8 * 1024 * 1024) -> Image.Image:
    """Decode a base64-encoded image and return a PIL Image. Enforce size limits."""
    try:
        header_sep = b64.find(',')
        if header_sep != -1:
            b64 = b64[header_sep + 1 :]

        decoded = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid base64 image")

    if len(decoded) > max_bytes:
        raise HTTPException(status_code=400, detail="image exceeds maximum allowed size")

    try:
        img = Image.open(io.BytesIO(decoded)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="decoded data is not a valid image")

    return img


def run_classifier(img: Image.Image, thresh: float = 0.5) -> Dict[str, Any]:
    model = app.state.classifier
    transform = app.state.transform
    device = app.state.device

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
        pred = int(torch.argmax(logits, dim=1).item())

    return {
        "predicted_label": pred,
        "probabilities": probs,
        "passed": (probs[1] >= thresh),
    }


def run_ocr_verification(img: Image.Image) -> Dict[str, Any]:
    ocr = app.state.ocr
    if ocr is None:
        return {"available": False, "message": "PaddleOCR not installed"}

    # PaddleOCR expects a file path or numpy array; convert to numpy BGR
    import numpy as np
    arr = np.array(img)[:, :, ::-1].copy()

    # Run OCR
    try:
        result = ocr.ocr(arr)
        texts = result[0]["rec_texts"] if result and isinstance(result, list) and len(result) > 0 else []
    except Exception as e:
        return {"available": True, "error": str(e)}

    # Simple marker-based verification as in demo_ocr
    import re
    text = " ".join(texts or [])
    text = re.sub(r"\s+", " ", text).strip().lower()
    markers = ["1", "2", "3", "4a", "4b", "4c", "4d", "5", "7", "8", "9"]
    found = {}
    for m in markers:
        if m.isdigit():
            pattern = rf"(?<!\d){m}\s*[\.\):,\-]"
        else:
            pattern = rf"\b{re.escape(m)}\b\s*[\.\):,\-]?"
        found[m] = re.search(pattern, text, flags=re.IGNORECASE) is not None

    missing = [m for m in markers if not found.get(m, False)]

    return {
        "available": True,
        "raw_text": texts,
        "found_markers": [m for m in markers if found.get(m, False)],
        "missing_markers": missing,
        "is_valid_format": len(missing) == 0,
    }


def run_face_verification(img: Image.Image) -> Dict[str, Any]:
    RF = app.state.retinaface
    if RF is None:
        return {"available": False, "message": "RetinaFace not installed"}

    import numpy as np
    import cv2

    # Convert to BGR and write temporary buffer for RetinaFace (it accepts path or ndarray?)
    arr = np.array(img)[:, :, ::-1].copy()

    # Save to a temporary in-memory file for compatibility with some RetinaFace wrappers
    # But RetinaFace.detect_faces can take a path; to avoid writing file, try passing ndarray
    try:
        detections_raw = RF.detect_faces(arr)
    except Exception:
        # Fallback: write to temp file
        import tempfile
        _, tmp = tempfile.mkstemp(suffix='.png')
        cv2.imwrite(tmp, arr)
        detections_raw = RF.detect_faces(tmp)
        try:
            os.remove(tmp)
        except Exception:
            pass

    faces = []
    if isinstance(detections_raw, dict):
        for key, det in detections_raw.items():
            score = float(det.get("score", 0.0))
            x1, y1, x2, y2 = det["facial_area"]
            faces.append({"score": score, "box": [int(x1), int(y1), int(x2), int(y2)]})

    # Basic verification: exactly one face with reasonable area
    h, w = arr.shape[:2]
    img_area = float(h * w)

    if len(faces) == 0:
        return {"available": True, "ok": False, "reason": "no_faces", "num_faces": 0, "faces": faces}
    if len(faces) > 1:
        return {"available": True, "ok": False, "reason": "multiple_faces", "num_faces": len(faces), "faces": faces}

    box = faces[0]["box"]
    fa = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    rel_area = fa / img_area if img_area > 0 else 0.0
    if rel_area < 0.02:
        return {"available": True, "ok": False, "reason": "face_too_small", "num_faces": 1, "faces": faces}
    if rel_area > 0.6:
        return {"available": True, "ok": False, "reason": "face_too_large", "num_faces": 1, "faces": faces}

    return {"available": True, "ok": True, "reason": "single_face_ok", "num_faces": 1, "faces": faces}


@app.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest):
    # Decode and validate image
    img = await run_in_threadpool(decode_base64_image, req.image_base64)

    # Stage 1: Binary classifier
    binary = await run_in_threadpool(run_classifier, img, req.thresh_binary)

    # If classifier rejects, short-circuit but still return OCR/face attempts
    ocr = await run_in_threadpool(run_ocr_verification, img)
    face = await run_in_threadpool(run_face_verification, img)

    ok = bool(binary.get("passed", False) and ocr.get("is_valid_format", False) and face.get("ok", False))

    return VerifyResponse(ok=ok, binary=binary, ocr=ocr, face=face)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
