import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Skin Analyzer Pro", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.hero-title {
    font-size: 2.2rem; font-weight: 600;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 4px;
}
.hero-sub { text-align:center; color:#64748b; font-size:0.95rem; margin-bottom:2rem; }
.metric-card {
    background:#f8fafc; border:1px solid #e2e8f0;
    border-radius:12px; padding:16px 20px; margin-bottom:12px;
}
.metric-label { font-size:0.78rem; color:#64748b; font-weight:500;
    letter-spacing:0.05em; text-transform:uppercase; }
.metric-value { font-size:1.6rem; font-weight:600; color:#1e293b;
    font-family:'DM Mono',monospace; }
.metric-bar-bg { background:#e2e8f0; border-radius:4px; height:6px; margin-top:8px; }
.metric-bar    { border-radius:4px; height:6px; }
.status-badge  { display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:0.75rem; font-weight:500; margin-left:8px; }
.badge-low  { background:#dcfce7; color:#15803d; }
.badge-mid  { background:#fef9c3; color:#92400e; }
.badge-high { background:#fee2e2; color:#b91c1c; }
.remedy-card {
    background:linear-gradient(135deg,#f0fdf4,#f8fafc);
    border:1px solid #bbf7d0; border-radius:12px;
    padding:18px 22px; margin-top:8px;
}
.remedy-title { font-size:1rem; font-weight:600; color:#15803d; margin-bottom:6px; }
.remedy-text  { font-size:0.88rem; color:#374151; line-height:1.6; }
.disclaimer-box {
    background:#fffbeb; border:1px solid #fcd34d; border-radius:10px;
    padding:14px 18px; margin-top:24px;
    font-size:0.83rem; color:#78350f; line-height:1.5;
}
.skin-tone-chip { display:inline-block; padding:4px 12px; border-radius:20px;
    font-size:0.8rem; font-weight:500; border:1.5px solid; }
.section-header { font-size:0.7rem; font-weight:600; letter-spacing:0.12em;
    text-transform:uppercase; color:#94a3b8; margin:20px 0 10px; }
</style>
""", unsafe_allow_html=True)

# ── Load Haar Cascade (bundled with opencv — no extra files needed) ──
@st.cache_resource(show_spinner=False)
def load_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        return None
    return detector

IMAGE_MAX_DIM = 1280

FITZPATRICK_PROFILES = {
    "Type I-II (Very Fair / Fair)": {
        "redness_hsv_low1":  np.array([0,   25, 120], dtype=np.uint8),
        "redness_hsv_high1": np.array([12, 180, 255], dtype=np.uint8),
        "redness_hsv_low2":  np.array([168, 25, 120], dtype=np.uint8),
        "redness_hsv_high2": np.array([180, 180, 255], dtype=np.uint8),
        "oil_threshold": 220,
        "description": "Baseline thresholds. Redness presents as pink-red.",
        "chip_color": "#fde8e8", "chip_border": "#f87171", "chip_text": "#991b1b"
    },
    "Type III-IV (Medium / Olive)": {
        "redness_hsv_low1":  np.array([0,   40,  80], dtype=np.uint8),
        "redness_hsv_high1": np.array([15, 220, 255], dtype=np.uint8),
        "redness_hsv_low2":  np.array([165, 40,  80], dtype=np.uint8),
        "redness_hsv_high2": np.array([180, 220, 255], dtype=np.uint8),
        "oil_threshold": 210,
        "description": "Adjusted for warmer undertones. Redness may appear deeper.",
        "chip_color": "#fef3c7", "chip_border": "#f59e0b", "chip_text": "#78350f"
    },
    "Type V-VI (Brown / Deep)": {
        "redness_hsv_low1":  np.array([0,   60,  60], dtype=np.uint8),
        "redness_hsv_high1": np.array([20, 255, 220], dtype=np.uint8),
        "redness_hsv_low2":  np.array([160, 60,  60], dtype=np.uint8),
        "redness_hsv_high2": np.array([180, 255, 220], dtype=np.uint8),
        "oil_threshold": 195,
        "description": "Wider hue range for deeper tones. Pigmentation analysis enhanced.",
        "chip_color": "#fdf2f8", "chip_border": "#a855f7", "chip_text": "#6b21a8"
    }
}

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="hero-title">&#128300; Skin Analyzer Pro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Computer vision skin surface analysis '
    '&middot; Fitzpatrick-calibrated &middot; No AI guesswork</div>',
    unsafe_allow_html=True
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    skin_tone_key = st.selectbox(
        "Fitzpatrick Skin Type",
        list(FITZPATRICK_PROFILES.keys()),
        help="Select the closest match to your skin tone."
    )
    profile = FITZPATRICK_PROFILES[skin_tone_key]
    st.markdown(
        f'<div style="margin-top:8px;padding:10px 12px;background:#f8fafc;'
        f'border-radius:8px;font-size:0.8rem;color:#475569;line-height:1.5;">'
        f'<b>Calibration:</b> {profile["description"]}</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    show_masks     = st.checkbox("Show raw detection masks", value=False)
    blur_threshold = st.slider("Sharpness threshold", 30, 150, 65,
        help="Lower = accept blurrier images.")

# ── Upload ────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a clear, front-facing photo (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("📷 Upload a photo above to begin analysis.")
    st.markdown(
        "**Tips for best results:**\n"
        "- Natural lighting or a ring light\n"
        "- Face the camera directly\n"
        "- No heavy filters or makeup\n"
        "- Minimum 720p resolution"
    )
    st.stop()

# ── Decode & Resize ───────────────────────────────────────────
uploaded_file.seek(0)
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img_bgr is None:
    st.error("Could not decode the image. Please try a different file.")
    st.stop()

h, w = img_bgr.shape[:2]
if max(h, w) > IMAGE_MAX_DIM:
    scale   = IMAGE_MAX_DIM / max(h, w)
    img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    h, w    = img_bgr.shape[:2]

# ── Sharpness Check ───────────────────────────────────────────
gray       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

if blur_score < blur_threshold:
    st.error(
        f"Image too blurry (score: {blur_score:.1f}, minimum: {blur_threshold}). "
        "Please retake in better lighting."
    )
    st.stop()

# ── Face Detection via Haar Cascade ──────────────────────────
detector = load_detector()
if detector is None:
    st.error("Face detector could not be loaded. Please reboot the app.")
    st.stop()

faces = detector.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5,
    minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE
)

if len(faces) == 0:
    st.error(
        "No face detected. Ensure your full face is visible, "
        "well-lit, and facing the camera directly."
    )
    st.stop()

# Use largest detected face
x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])

# Build face mask from bounding box with slight inset
pad_x = int(fw * 0.08)
pad_y = int(fh * 0.08)
x1 = max(x + pad_x, 0);  y1 = max(y + pad_y, 0)
x2 = min(x + fw - pad_x, w); y2 = min(y + fh - pad_y, h)

face_mask = np.zeros((h, w), dtype=np.uint8)
face_mask[y1:y2, x1:x2] = 255
total_face_px = max(int(np.sum(face_mask == 255)), 1)

# ── Pixel Analysis ────────────────────────────────────────────
hsv    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
kernel = np.ones((3, 3), np.uint8)

# 1. Redness
mask_r1  = cv2.inRange(hsv, profile["redness_hsv_low1"], profile["redness_hsv_high1"])
mask_r2  = cv2.inRange(hsv, profile["redness_hsv_low2"], profile["redness_hsv_high2"])
raw_red  = cv2.morphologyEx(cv2.bitwise_or(mask_r1, mask_r2), cv2.MORPH_OPEN, kernel)
red_mask = cv2.bitwise_and(raw_red, face_mask)

# 2. Oiliness
_, raw_oil = cv2.threshold(gray, profile["oil_threshold"], 255, cv2.THRESH_BINARY)
raw_oil    = cv2.morphologyEx(raw_oil, cv2.MORPH_OPEN, kernel)
oil_mask   = cv2.bitwise_and(raw_oil, face_mask)

# 3. Pigmentation
hue_ch     = hsv[:, :, 0].astype(np.float32)
hue_masked = np.where(face_mask == 255, hue_ch, np.nan)
with np.errstate(all="ignore"):
    hue_std = float(np.nanstd(hue_masked))
pigment_score = min(int((hue_std / 40.0) * 100), 100)

# 4. Texture roughness
lap           = cv2.Laplacian(gray, cv2.CV_64F)
lap_vals      = lap[face_mask == 255]
texture_raw   = float(np.std(lap_vals)) if len(lap_vals) > 0 else 0.0
texture_score = min(int((texture_raw / 80.0) * 100), 100)

# Final scores
redness_score  = min(int((np.sum(red_mask  > 0) / total_face_px) * 200), 100)
oiliness_score = min(int((np.sum(oil_mask  > 0) / total_face_px) * 350), 100)

# ── Overlay ───────────────────────────────────────────────────
overlay               = img_bgr.copy()
overlay[red_mask > 0] = [0,   0,   220]
overlay[oil_mask > 0] = [220, 140,   0]
# Draw face bounding box on overlay
cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 200, 100), 2)
final_overlay = cv2.addWeighted(overlay, 0.28, img_bgr, 0.72, 0)
cv2.rectangle(final_overlay, (x1, y1), (x2, y2), (100, 200, 100), 2)
final_rgb     = cv2.cvtColor(final_overlay, cv2.COLOR_BGR2RGB)

# ── Layout ────────────────────────────────────────────────────
col_img, col_results = st.columns([1.1, 1])

with col_img:
    st.markdown('<div class="section-header">📸 Analysis Overlay</div>', unsafe_allow_html=True)
    st.image(final_rgb,
             caption="🟢 Face zone  |  🔴 Inflammation  |  🟠 Sebum/shine",
             use_container_width=True)

    if show_masks:
        with st.expander("Raw detection masks"):
            mc1, mc2 = st.columns(2)
            with mc1:
                st.image(red_mask, caption="Redness mask", use_container_width=True)
            with mc2:
                st.image(oil_mask, caption="Oil/shine mask", use_container_width=True)

    st.markdown(
        f'<div style="margin-top:10px;font-size:0.78rem;color:#64748b;">'
        f'Image: {w}&times;{h}px &nbsp;&middot;&nbsp; '
        f'Sharpness: <b>{blur_score:.1f}</b> &nbsp;&middot;&nbsp; '
        f'Face: <b>&#9989; Detected</b></div>',
        unsafe_allow_html=True
    )

with col_results:
    st.markdown('<div class="section-header">📊 Skin Parameters</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="margin-bottom:14px;">'
        f'<span class="skin-tone-chip" style="background:{profile["chip_color"]};'
        f'border-color:{profile["chip_border"]};color:{profile["chip_text"]};">'
        f'&#127912; {skin_tone_key}</span></div>',
        unsafe_allow_html=True
    )

    def badge(s):
        if s < 30:   return '<span class="status-badge badge-low">Low</span>'
        elif s < 60: return '<span class="status-badge badge-mid">Moderate</span>'
        else:        return '<span class="status-badge badge-high">Elevated</span>'

    def bar_color(s):
        if s < 30:   return "#22c55e"
        elif s < 60: return "#f59e0b"
        else:        return "#ef4444"

    for label, score, tooltip in [
        ("🔴 Redness Index",       redness_score,  "Vascular inflammation signal"),
        ("💧 Sebum / Oiliness",    oiliness_score, "High-reflectance highlight density"),
        ("🎨 Pigmentation Spread", pigment_score,  "Hue std-dev — uneven tone indicator"),
        ("🪨 Texture Roughness",   texture_score,  "Laplacian edge variance"),
    ]:
        st.markdown(
            f'<div class="metric-card" title="{tooltip}">'
            f'<div class="metric-label">{label} {badge(score)}</div>'
            f'<div class="metric-value">{score}'
            f'<span style="font-size:0.9rem;color:#94a3b8">%</span></div>'
            f'<div class="metric-bar-bg">'
            f'<div class="metric-bar" style="width:{score}%;background:{bar_color(score)};"></div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-header">🌿 Evidence-Based Suggestions</div>',
                unsafe_allow_html=True)

    dominant = max(
        [("redness", redness_score), ("oiliness", oiliness_score),
         ("pigmentation", pigment_score), ("texture", texture_score)],
        key=lambda x: x[1]
    )

    if dominant[0] == "redness" and redness_score >= 25:
        st.markdown("""<div class="remedy-card">
            <div class="remedy-title">Calm Vascular Irritation</div>
            <div class="remedy-text">Elevated redness indicates potential inflammation.
            Consider a fragrance-free, ceramide-rich moisturizer.
            Clinically evidenced: <b>Niacinamide (4&#8211;5%)</b>,
            <b>Azelaic acid (10%)</b>, <b>Centella Asiatica extract</b>.
            Avoid harsh physical exfoliants.</div></div>""", unsafe_allow_html=True)
    elif dominant[0] == "oiliness" and oiliness_score >= 25:
        st.markdown("""<div class="remedy-card">
            <div class="remedy-title">Regulate Sebum Production</div>
            <div class="remedy-text">High shine suggests overactive sebaceous glands.
            A gentle foaming cleanser and oil-free moisturizer help.
            Studied ingredients: <b>Niacinamide (2&#8211;5%)</b>,
            <b>Salicylic acid (0.5&#8211;2%)</b>, <b>Zinc PCA</b>.</div></div>""",
            unsafe_allow_html=True)
    elif dominant[0] == "pigmentation" and pigment_score >= 30:
        st.markdown("""<div class="remedy-card">
            <div class="remedy-title">Even Skin Tone</div>
            <div class="remedy-text">Uneven hue may indicate hyperpigmentation or sun damage.
            Daily SPF 30+ is essential. Strong evidence:
            <b>Vitamin C (10&#8211;20%)</b>, <b>Alpha Arbutin (2%)</b>, <b>Kojic acid</b>.
            Consult a dermatologist for persistent dark spots.</div></div>""",
            unsafe_allow_html=True)
    else:
        st.markdown("""<div class="remedy-card">
            <div class="remedy-title">Maintain Barrier Health</div>
            <div class="remedy-text">Skin parameters appear within normal ranges.
            Focus on: gentle cleanser, <b>ceramide or hyaluronic acid moisturizer</b>,
            and daily <b>broad-spectrum SPF 30+</b>.</div></div>""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    &#9888;&#65039; <b>Medical Disclaimer:</b> This tool performs non-clinical computer vision
    analysis only. It does <b>not</b> diagnose any medical condition or replace a qualified
    dermatologist. If you notice unusual moles, lesions, or any changing skin abnormality,
    consult a licensed dermatologist promptly.
</div>
""", unsafe_allow_html=True)
