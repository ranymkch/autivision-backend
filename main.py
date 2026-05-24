"""
============================================================
ASD Detection API — MobileNetV2 (FastAPI)
============================================================
Objectif : Classifier des images faciales d'enfants comme
           autistes ou non-autistes (ASD vs. TD).
============================================================
"""

import cv2
import hashlib
import io
import math
import os
import logging
import time
import urllib.request
from contextlib import asynccontextmanager
from typing import Any, Optional

import mediapipe as mp
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from packaging import version as pkgver
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

# ─────────────────────────────────────────────
# Configuration du logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Chargement des variables d'environnement (.env)
# ─────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ─────────────────────────────────────────────
# Constantes & paramètres du modèle
# ─────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH: str = os.getenv(
    "MODEL_PATH",
    os.path.join(_HERE, "best_mobilenetv2.h5"),
)
IMG_SIZE: tuple[int, int] = (224, 224)
DECISION_THRESHOLD: float = float(
    os.getenv("DECISION_THRESHOLD", "0.50")
)
TTA_N: int = int(os.getenv("TTA_N", "5"))   # augmented inference passes (0 = disabled)

# Ordre alphabétique issu de Keras flow_from_directory
# 0 → autistic, 1 → non_autistic
CLASS_NAMES: list[str] = ["autistic", "non_autistic"]

# Forcer l'exécution sur CPU uniquement
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")


# ─────────────────────────────────────────────
# Constantes MediaPipe
# ─────────────────────────────────────────────
MP_VERSION = pkgver.parse(mp.__version__)
MP_NEW_API = MP_VERSION >= pkgver.parse('0.10.0')

MP_LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MP_FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172,  58, 132,
     93, 234, 127, 162,  21,  54, 103,  67, 109,
]

_MP_MODEL_PATH = "face_landmarker.task"
_MP_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


# ─────────────────────────────────────────────
# État global — modèle chargé une seule fois
# ─────────────────────────────────────────────

class AppState:
    model: Optional[tf.keras.Model] = None
    model_loaded: bool = False
    load_error: Optional[str] = None
    startup_time: Optional[float] = None
    mp_detector: Optional[Any] = None
    mp_api: Optional[str] = None

state = AppState()


# ─────────────────────────────────────────────
# Initialisation MediaPipe
# ─────────────────────────────────────────────

def _init_mediapipe_detector(min_detection_conf: float = 0.4):
    """Initialise le détecteur MediaPipe (Tasks API ou legacy selon version)."""
    if MP_NEW_API:
        try:
            from mediapipe.tasks.python import vision as mpv
            from mediapipe.tasks.python.core import base_options as mpb

            if not os.path.exists(_MP_MODEL_PATH):
                logger.info("Téléchargement du modèle FaceLandmarker (~4 MB)…")
                urllib.request.urlretrieve(_MP_MODEL_URL, _MP_MODEL_PATH)

            opts = mpv.FaceLandmarkerOptions(
                base_options=mpb.BaseOptions(model_asset_path=_MP_MODEL_PATH),
                num_faces=1,
                min_face_detection_confidence=min_detection_conf,
                min_face_presence_confidence=min_detection_conf,
                min_tracking_confidence=min_detection_conf,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            detector = mpv.FaceLandmarker.create_from_options(opts)
            logger.info(f"MediaPipe Tasks API (FaceLandmarker) initialisé — v{mp.__version__}")
            return detector, 'tasks'
        except Exception as exc:
            logger.warning(f"Tasks API échouée ({exc}), fallback legacy.")

    detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=min_detection_conf,
        refine_landmarks=True,
    )
    logger.info(f"MediaPipe Legacy API (face_mesh) initialisé — v{mp.__version__}")
    return detector, 'legacy'


# ─────────────────────────────────────────────
# Lifespan : chargement au démarrage
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    logger.info("Démarrage de l'API ASD Detection (MobileNetV2)")

    if os.path.exists(MODEL_PATH):
        try:
            logger.info(f"Chargement du modèle depuis : {MODEL_PATH}")
            state.model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,
            )
            state.model_loaded = True
            logger.info(f"Modèle chargé en {time.perf_counter() - t0:.2f}s — CPU uniquement")
        except Exception as exc:
            state.load_error = str(exc)
            logger.error(f"Erreur de chargement du modèle : {exc}")
    else:
        state.load_error = f"Fichier modèle introuvable : {MODEL_PATH}"
        logger.warning(state.load_error)

    state.mp_detector, state.mp_api = _init_mediapipe_detector()

    state.startup_time = time.perf_counter() - t0

    yield

    logger.info("Arrêt de l'API.")
    state.model = None
    state.model_loaded = False
    if state.mp_detector is not None:
        try:
            state.mp_detector.close()
        except Exception:
            pass


# ─────────────────────────────────────────────
# Application FastAPI
# ─────────────────────────────────────────────

app = FastAPI(
    title="ASD Detection API — MobileNetV2",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Schémas Pydantic
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Pydantic Response Model
# ─────────────────────────────────────────────

class PredictionResponse(BaseModel):
    prediction: str          # "autistic" or "non_autistic"
    asd_risk: float          # P(autistic) ∈ [0,1]
    asd_risk_pct: str        # ex. "86.4%"
    threshold_used: float
    model: str  


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    load_error: Optional[str]
    startup_time_s: Optional[float]
    device: str
    img_size: list[int]
    threshold: float


# ─────────────────────────────────────────────
# Helpers géométrie MediaPipe
# ─────────────────────────────────────────────

def _eye_centre(lm_list, indices, img_w, img_h):
    xs = [lm_list[i].x * img_w for i in indices]
    ys = [lm_list[i].y * img_h for i in indices]
    return (float(np.mean(xs)), float(np.mean(ys)))

def _oval_bbox(lm_list, indices, img_w, img_h):
    xs = [lm_list[i].x * img_w for i in indices]
    ys = [lm_list[i].y * img_h for i in indices]
    return (min(xs), min(ys), max(xs), max(ys))

def _eye_angle_deg(left_eye, right_eye):
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    return math.degrees(math.atan2(dy, dx))

def _build_rotation_matrix(img_shape, angle_deg, pivot=None):
    h, w = img_shape[:2]
    if pivot is None:
        pivot = (w / 2.0, h / 2.0)
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    px, py = float(pivot[0]), float(pivot[1])
    m00, m01 = cos_a, sin_a
    m02 = px - px * cos_a - py * sin_a
    m10, m11 = -sin_a, cos_a
    m12 = py + px * sin_a - py * cos_a
    corners = [(0., 0.), (float(w), 0.), (float(w), float(h)), (0., float(h))]
    rot_xs = [m00*x + m01*y + m02 for x, y in corners]
    rot_ys = [m10*x + m11*y + m12 for x, y in corners]
    x_min, y_min = min(rot_xs), min(rot_ys)
    new_w = int(math.ceil(max(rot_xs) - x_min))
    new_h = int(math.ceil(max(rot_ys) - y_min))
    m02 -= x_min
    m12 -= y_min
    M = np.array([[m00, m01, m02], [m10, m11, m12]], dtype=np.float64)
    return M, (new_w, new_h)

def _transform_point(pt, M):
    x, y = float(pt[0]), float(pt[1])
    return (M[0,0]*x + M[0,1]*y + M[0,2], M[1,0]*x + M[1,1]*y + M[1,2])

def _transform_bbox(bbox, M):
    x1, y1, x2, y2 = bbox
    corners = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
    rc = [_transform_point(c, M) for c in corners]
    return (min(p[0] for p in rc), min(p[1] for p in rc),
            max(p[0] for p in rc), max(p[1] for p in rc))

def _crop_face_region(img, bbox, padding_ratio=0.30, target_size=(224, 224)):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    pad_x, pad_y = int(bw * padding_ratio), int(bh * padding_ratio)
    H, W = img.shape[:2]
    crop = img[max(0, y1-pad_y):min(H, y2+pad_y),
               max(0, x1-pad_x):min(W, x2+pad_x)]
    if crop.size == 0:
        return None
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_LANCZOS4)


# ─────────────────────────────────────────────
# Prétraitement de l'image
# ─────────────────────────────────────────────

def _get_face_crop(image_bytes: bytes) -> np.ndarray:
    """
    Étapes 1-5 du pipeline : retourne un crop facial brut uint8 (224, 224, 3).

    1. Décodage bytes → RGB numpy array
    2. Détection des landmarks faciaux (Tasks API ou legacy)
    3. Calcul de l'angle d'alignement des yeux
    4. Rotation canvas-expanding pour niveler la ligne des yeux
    5. Crop avec padding adaptatif (30%) + resize 224×224

    Fallback : si aucun visage détecté, resize direct 224×224 sans crop.
    """
    # 1. Décodage
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.load()
        img_rgb = np.array(pil_img.convert('RGB'), dtype=np.uint8)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Fichier invalide. Formats acceptés : JPEG, PNG, WEBP, BMP.",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur ouverture image : {exc}")

    # 2. Détection MediaPipe
    crop = None
    if state.mp_detector is not None:
        try:
            h, w = img_rgb.shape[:2]

            if state.mp_api == 'tasks':
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result = state.mp_detector.detect(mp_img)
                lm_lists = result.face_landmarks if result.face_landmarks else []
            else:  # legacy
                result = state.mp_detector.process(img_rgb)
                lm_lists = (
                    [f.landmark for f in result.multi_face_landmarks]
                    if result.multi_face_landmarks else []
                )

            if lm_lists:
                lm_list   = lm_lists[0]
                bbox_orig = _oval_bbox(lm_list, MP_FACE_OVAL_IDX, w, h)
                left_eye  = _eye_centre(lm_list, MP_LEFT_EYE_IDX,  w, h)
                right_eye = _eye_centre(lm_list, MP_RIGHT_EYE_IDX, w, h)
                angle     = _eye_angle_deg(left_eye, right_eye)
                pivot     = ((left_eye[0]+right_eye[0])/2.0,
                             (left_eye[1]+right_eye[1])/2.0)
                M, new_size = _build_rotation_matrix(img_rgb.shape, angle, pivot)
                rotated_img = cv2.warpAffine(
                    img_rgb, M, new_size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                bbox_rot = _transform_bbox(bbox_orig, M)
                crop     = _crop_face_region(rotated_img, bbox_rot,
                                             padding_ratio=0.30,
                                             target_size=IMG_SIZE)

        except Exception as exc:
            logger.warning(f"MediaPipe échoué, fallback resize direct : {exc}")

    # 3. Aucun visage détecté — on rejette la requête.
    # Le modèle n'a jamais été entraîné sur des images non-croppées : un resize
    # direct produirait des prédictions non-fiables sans avertissement.
    if crop is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Aucun visage détecté dans l'image. "
                "Veuillez fournir une photo frontale nette avec le visage bien visible."
            ),
        )

    return crop  # uint8 (224, 224, 3)


def _augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Augmentation aléatoire identique au TTA du notebook d'entraînement.
    Entrée/sortie : uint8 RGB (224, 224, 3).

    Paramètres (fidèles au notebook) :
      rotation ±25°, shift ±12%, zoom [0.80-1.20],
      flip horizontal 50%, brightness [0.75-1.25], channel shift ±20.
    """
    h, w = img.shape[:2]

    # Rotation + zoom + translation (une seule warpAffine)
    angle  = float(rng.uniform(-25.0,  25.0))
    scale  = float(rng.uniform( 0.80,   1.20))
    tx     = float(rng.uniform(-0.12,   0.12)) * w
    ty     = float(rng.uniform(-0.12,   0.12)) * h

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)  # fill_mode='nearest'

    # Shear ±10% (matches notebook TTA shear_range=0.10)
    shear = float(rng.uniform(-0.10, 0.10))
    shear_M = np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    img = cv2.warpAffine(img, shear_M, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    # Horizontal flip
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)

    # Brightness — applied in HSV V-channel (matches Keras ImageDataGenerator
    # brightness_range which also operates on the V channel, not RGB channels)
    bright = float(rng.uniform(0.75, 1.25))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * bright, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Channel shift ±20
    shift = rng.uniform(-20.0, 20.0, size=(1, 1, 3)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + shift, 0, 255).astype(np.uint8)

    return img


def _tta_predict(crop: np.ndarray) -> float:
    """
    Test-Time Augmentation : 1 pass propre (non augmenté) + TTA_N passes
    augmentées, score sigmoid moyenné sur TTA_N+1 passes au total.

    Inclure le pass propre en premier ancre la moyenne sur la prédiction
    réelle et empêche que TTA_N passes augmentées agressives ne fassent
    basculer un cas limite (ex. prob ≈ 0.55 → autistic par erreur).

    Si TTA_N == 0 : un seul passage non augmenté (comportement déterministe).

    MobileNetV2 attend des pixels normalisés dans [-1, 1].
    """
    def _preprocess(img: np.ndarray) -> np.ndarray:
        return tf.keras.applications.mobilenet_v2.preprocess_input(img.astype(np.float32))

    arr_clean = np.expand_dims(_preprocess(crop), 0)
    clean_score = float(state.model.predict(arr_clean, verbose=0)[0][0])

    if TTA_N == 0:
        return clean_score

    seed = int(hashlib.md5(crop.tobytes()).hexdigest()[:8], 16) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    scores: list[float] = [clean_score]   # anchor with unaugmented prediction
    for _ in range(TTA_N):
        aug = _augment(crop, rng)
        arr = np.expand_dims(_preprocess(aug), 0)
        scores.append(float(state.model.predict(arr, verbose=0)[0][0]))

    return float(np.mean(scores))


# ─────────────────────────────────────────────
# Label Prediction Logic
# ─────────────────────────────────────────────

def get_prediction_label(probability: float) -> str:
    """
    probability = P(non_autistic)

    If probability is high  -> non_autistic
    If probability is low   -> autistic
    """

    if probability >= DECISION_THRESHOLD:
        return "non_autistic"

    return "autistic"


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/", tags=["Général"])
async def root():
    return {
        "api": "ASD Detection API",
        "model": "MobileNetV2",
        "version": "2.1.0",
        "tta_passes": TTA_N,
        "endpoints": {
            "health": "/health",
            "predict": "/predict  [POST, multipart/form-data]",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """État de l'API et du modèle. Retourne 503 si le modèle est indisponible."""
    response = HealthResponse(
        status="ok" if state.model_loaded else "degraded",
        model_loaded=state.model_loaded,
        model_path=MODEL_PATH,
        load_error=state.load_error,
        startup_time_s=round(state.startup_time, 3) if state.startup_time else None,
        device="CPU (GPU désactivé)",
        img_size=list(IMG_SIZE),
        threshold=DECISION_THRESHOLD,
    )

    if not state.model_loaded:
        return JSONResponse(status_code=503, content=response.model_dump())

    return response


# ─────────────────────────────────────────────
# Predict Endpoint (CORRECTED)
# ─────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, tags=["Inférence"])
async def predict(file: UploadFile = File(..., description="Image faciale (JPEG/PNG)")):

    # 1. Model available?
    if not state.model_loaded or state.model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modèle non disponible : {state.load_error}",
        )

    # 2. MIME validation
    allowed_content_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
    }

    content_type = (file.content_type or "").lower()

    if content_type and content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Type non supporté : '{content_type}'. "
                f"Acceptés : {', '.join(allowed_content_types)}"
            ),
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Nom de fichier manquant.",
        )

    # 3. Read file bytes
    try:
        image_bytes = await file.read()

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Impossible de lire le fichier : {exc}",
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Le fichier est vide.",
        )

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Fichier trop grand ({len(image_bytes) / 1e6:.1f} Mo). "
                f"Maximum : 10 Mo."
            ),
        )

    # 4. Face crop + alignment
    logger.info(
        f"Inférence TTA×{TTA_N} : "
        f"{file.filename} ({len(image_bytes) / 1024:.1f} Ko)"
    )

    try:
        crop = _get_face_crop(image_bytes)

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(f"Erreur prétraitement : {exc}")

        raise HTTPException(
            status_code=500,
            detail=f"Erreur de prétraitement : {exc}",
        )

    # 5. Model inference
    try:
        t_infer = time.perf_counter()

        # probability = P(non_autistic)
        probability = _tta_predict(crop)

        infer_ms = (time.perf_counter() - t_infer) * 1000

        logger.info(
            f"P(non_autistic)={probability:.4f} | "
            f"{infer_ms:.1f}ms | "
            f"TTA×{TTA_N} | "
            f"{file.filename}"
        )

    except Exception as exc:
        logger.error(f"Erreur inférence : {exc}")

        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'inférence : {exc}",
        )

    # 6. Final interpretation

    # ASD risk = P(autistic)
    asd_risk = 1.0 - probability

    # Predicted class
    label = get_prediction_label(probability)

    logger.info(
        f"Prediction={label} | "
        f"ASD_RISK={asd_risk:.4f}"
    )

    return PredictionResponse(
        prediction=label,
        asd_risk=round(asd_risk, 6),
        asd_risk_pct=f"{asd_risk * 100:.1f}%",
        threshold_used=DECISION_THRESHOLD,
        model="MobileNetV2 (ASD fine-tuned)",
    )