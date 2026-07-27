import base64
import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

ONNX_PATH = os.environ.get("ONNX_MODEL_PATH", "sam2unet_conv_bghr.onnx")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SAM_SIZE = 1024   # x_sam input resolution
CNX_SIZE = 448    # x_cnx input resolution
OUT_SIZE = 352    # native model output resolution before resizing back

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if "CUDAExecutionProvider" in ort.get_available_providers() \
        else ["CPUExecutionProvider"]

    if not os.path.exists(ONNX_PATH):
        raise RuntimeError(
            f"ONNX model not found at '{ONNX_PATH}'. Set ONNX_MODEL_PATH env var or place the exported .onnx file next to app.py."
        )

    state["session"] = ort.InferenceSession(ONNX_PATH, providers=providers)
    state["providers"] = providers
    print(f"[app] Loaded {ONNX_PATH} with providers={providers}")
    yield
    state.clear()


app = FastAPI(title="SAM2-DEB-UNet Polyp Segmentation API", lifespan=lifespan)


def preprocess(img: Image.Image, size: int) -> np.ndarray:
    """Resize -> [0,1] -> normalize (ImageNet mean/std) -> NCHW float32."""
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None, ...]  # (1, 3, H, W)
    return np.ascontiguousarray(arr, dtype=np.float32)


def sigmoid_to_mask(prob: np.ndarray, orig_size: tuple[int, int], threshold: float) -> Image.Image:
    """prob: (1,1,H,W) in [0,1]. Resize to original (W,H) and threshold to 0/255."""
    prob_img = Image.fromarray((prob[0, 0] * 255).astype(np.uint8))
    prob_img = prob_img.resize(orig_size, Image.BILINEAR)
    mask = (np.asarray(prob_img) / 255.0 >= threshold).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def run_inference(img: Image.Image, threshold: float) -> tuple[Image.Image, dict]:
    orig_w, orig_h = img.size

    x_sam = preprocess(img, SAM_SIZE)
    x_cnx = preprocess(img, CNX_SIZE)

    t0 = time.perf_counter()
    outputs = state["session"].run(None, {"x_sam": x_sam, "x_cnx": x_cnx})
    latency_ms = (time.perf_counter() - t0) * 1000

    prob = outputs[0]  # (1,1,352,352) sigmoid probability
    mask_img = sigmoid_to_mask(prob, (orig_w, orig_h), threshold)

    polyp_ratio = float((np.asarray(mask_img) > 0).mean())
    stats = {
        "latency_ms": round(latency_ms, 2),
        "polyp_area_ratio": round(polyp_ratio, 4),
        "threshold": threshold,
        "original_size": [orig_w, orig_h],
    }
    return mask_img, stats


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": ONNX_PATH,
        "providers": state.get("providers", []),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    mask_img, stats = run_inference(img, threshold)

    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    headers = {
        "X-Latency-Ms": str(stats["latency_ms"]),
        "X-Polyp-Area-Ratio": str(stats["polyp_area_ratio"]),
    }
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)


@app.post("/predict/json")
async def predict_json(file: UploadFile = File(...), threshold: float = 0.5):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    mask_img, stats = run_inference(img, threshold)

    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({"mask_png_base64": mask_b64, **stats})
