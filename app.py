import streamlit as st
import cv2
import numpy as np
import warnings
from PIL import Image

# ── Mediapipe — top-level import with clear HF/Streamlit error guidance ─
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# ── Page Config (Must be the very first Streamlit call) ────────────────
st.set_page_config(
    page_title="Skin Analyzer Pro",
    page_icon="🔬",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://googleapis.com');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.hero-title {
    font-size: 2.2rem; font-weight: 600;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 4px;
}
.hero-sub {
    text-align: center; color: #64748b;
    font-size: 0.95rem; margin-bottom: 2rem;
}
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
}
.metric-label {
    font-size: 0.78rem; color: #64748b; font-weight: 500;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.metric-value {
    font-size: 1.6rem; font-weight: 600; color: #1e293b;
    font-family: 'DM Mono', monospace;
}
.metric-bar-bg { background: #e2e8f0; border-radius: 4px; height: 6px; margin-top: 8px; }
.metric-bar { border-radius: 4px; height: 6px; }
.status-badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 500; margin-left: 8px;
}
.badge-low { background: #dcfce7; color: #15803d; }
.badge-mid { background: #fef9c3; color: #92400e; }
.badge-high { background: #fee2e2; color: #b91c1c; }
.remedy-card {
    background: linear-gradient(135deg, #f0fdf4, #f8fafc);
    border: 1px solid #bbf7d0; border-radius: 12px;
    padding: 18px 22px; margin-top: 8px;
}
.remedy-title { font-size: 1rem; font-weight: 600; color: #15803d; margin-bottom: 6px; }
.remedy-text { font-size: 0.88rem; color: #374151; line-height: 1.6; }
.disclaimer-box {
    background: #fffbeb; border: 1px solid #fcd34d; border-radius: 10px;
    padding: 14px 18px; margin-top: 24px;
    font-size: 0.83rem; color: #78350f; line-height: 1.5;
}
.skin-tone-chip {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 500; border: 1.5px solid;
}
.section-header {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #94a3b8; margin: 20px 0 10px;
}
.col-gap-right { padding-right: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Hard-stop if mediapipe missing ───────────────────────────
if not _MP_AVAILABLE:
    st.error(
        "**MediaPipe not found.** Verify that it is listed in your "
        "`requirements.txt` file and wait for Streamlit to rebuild."
    )
    st.stop()

# ── Cache FaceMesh — initialize ONCE across all reruns ────────
@st.cache_resource(show_spinner=False)
def load_face_mesh():
    mp_fm = mp.solutions.face_mesh
    return mp_fm.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

# ── Constants ─────────────────────────────────────────────────
IMAGE_MAX_DIM = 1280

# ── Fitzpatrick Profiles ──────────────────────────────────────
FITZPATRICK_PROFILES = {
    "Type I-II (Very Fair / Fair)": {
        "redness_hsv_low1": np.array([0, 25, 120], dtype=np.uint8),
        "redness_hsv_high1": np.array([12, 180, 255], dtype=np.uint8),
        "redness_hsv_low2": np.array([168, 25, 120], dtype=np.uint8),
        "redness_hsv_high2": np.array([180, 180, 255], dtype=np.uint8),
        "oil_threshold": 220,
        "description": "Baseline thresholds. Redness presents as pink-red.",
        "chip_color": "#fde8e8", "chip_border": "#f87171", "chip_text": "#991b1b"
    },
    "Type III-IV (Medium / Olive)": {
        "redness_hsv_low1": np.array([0, 40, 80], dtype=np.uint8),
        "redness_hsv_high1": np.array([15, 220, 255], dtype=np.uint8),
        "redness_hsv_low2": np.array([165, 40, 80], dtype=np.uint8),
        "redness_hsv_high2": np.array([180, 220, 255], dtype=np.uint8),
        "oil_threshold": 210,
        "description": "Adjusted for warmer undertones. Redness may appear deeper.",
        "chip_color": "#fef3c7", "chip_border": "#f59e0b", "chip_text": "#78350f"
    },
    "Type V-VI (Brown / Deep)": {
        "redness_hsv_low1": np.array([0, 60, 60], dtype=np.uint8),
        "redness_hsv_high1": np.array([20, 255, 220], dtype=np.uint8),
        "redness_hsv_low2": np.array([160, 60, 60], dtype=np.uint8),
        "redness_hsv_high2": np.array([180, 255, 220], dtype=np.uint8),
        "oil_threshold": 195,
        "description": "Wider hue range for deeper tones. Pigmentation analysis enhanced.",
        "chip_color": "#fdf2f8", "chip_border": "#a855f7", "chip_text": "#6b21a8"
    }
}

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🔬 Skin Analyzer Pro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Computer vision skin surface analysis &middot; '
    'Fitzpatrick-calibrated &middot; No AI guesswork</div>',
    unsafe_allow_html=True
)

# ── Sidebar Settings ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    skin_tone_key = st.selectbox(
        "Fitzpatrick Skin Type",
        list(FITZPATRICK_PROFILES.keys()),
        help="Select the closest match to your skin tone for accurate calibration."
    )
    profile = FITZPATRICK_PROFILES[skin_tone_key]
    st.markdown(
        f'<div style="margin-top:8px;padding:10px 12px;background:#f8fafc;'
        f'border-radius:8px;font-size:0.8rem;color:#475569;line-height:1.5;">'
        f'<b>Calibration:</b> {profile["description"]}</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    show_masks = st.checkbox("Show raw detection masks", value=False)
    blur_threshold = st.slider(
        "Sharpness threshold", 30, 150, 65,
        help="Lower = accept blurrier images. 65 is recommended."
    )

# ── Upload ────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a clear, front-facing photo (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("📷 Upload a photo above to begin analysis.")
    st.markdown(
        "**Tips for best results:**\n"
        "- Use natural lighting or a ring light\n"
        "- Face the camera directly\n"
        "- Avoid heavy filters or makeup\n"
        "- Minimum 720p resolution recommended"
    )
    st.stop()

# ── Decode Image ──────────────────────────────────────────────
uploaded_file.seek(0)
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img_bgr is None:
    st.error("❌ Could not decode the image. Please try a different file.")
    st.stop()

# Resize oversized images to safeguard memory limits on Cloud host
h, w = img_bgr.shape[:2]
if max(h, w) > IMAGE_MAX_DIM:
    scale = IMAGE_MAX_DIM / max(h, w)
    img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = img_bgr.shape[:2]

# ── Sharpness Check ───────────────────────────────────────────
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
if blur_score < blur_threshold:
    st.error(
        f"❌ Image too blurry (score: {blur_score:.1f}, minimum: {blur_threshold}). "
        "Please retake in brighter, steadier conditions."
    )
    st.stop()

# ── Face Mesh Detection ───────────────────────────────────────
face_mesh = load_face_mesh()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results = face_mesh.process(img_rgb)

if not results.multi_face_landmarks:
    st.error(
        "❌ No face detected. Ensure your full face is visible, "
        "well-lit, and facing the camera directly."
    )
    st.stop()

# FIX: Target index list wrapper to avoid execution crash
landmarks = results.multi_face_landmarks[0].landmark

# ── Face Zone Landmark Indices ────────────────────────────────
T_ZONE_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]
CHEEK_IDX = [
    205, 203, 98, 97, 2, 327, 326, 425, 423, 427, 207,
    116, 111, 117, 118, 101, 212, 214, 192, 210, 211,
    345, 340, 346, 347, 330, 432, 434, 416, 430, 431
]

def lm_to_px(indices):
    return np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices],
        dtype=np.int32
    )

t_zone_pts = lm_to_px(T_ZONE_IDX)
cheek_pts = lm_to_px(CHEEK_IDX)
face_mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(face_mask, [t_zone_pts, cheek_pts], 255)
total_face_px = max(int(np.sum(face_mask == 255)), 1)

# ── Pixel Analysis ────────────────────────────────────────────
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
kernel = np.ones((3, 3), np.uint8)

# 1. Redness
mask_r1 = cv2.inRange(hsv, profile["redness_hsv_low1"], profile["redness_hsv_high1"])
mask_r2 = cv2.inRange(hsv, profile["redness_hsv_low2"], profile["redness_hsv_high2"])
raw_red = cv2.morphologyEx(cv2.bitwise_or(mask_r1, mask_r2), cv2.MORPH_OPEN, kernel)
red_mask = cv2.bitwise_and(raw_red, face_mask)

# 2. Oiliness / sebum shine
_, raw_oil = cv2.threshold(gray, profile["oil_threshold"], 255, cv2.THRESH_BINARY)
raw_oil = cv2.morphologyEx(raw_oil, cv2.MORPH_OPEN, kernel)
oil_mask = cv2.bitwise_and(raw_oil, face_mask)

