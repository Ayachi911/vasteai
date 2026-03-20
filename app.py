import streamlit as st
from PIL import Image
import numpy as np
import io
import time
import pickle
import os

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="VasteAI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── LOAD MODEL ──
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

# ── FEATURE EXTRACTION (must match training) ──
def extract_features(img: Image.Image) -> np.ndarray:
    from skimage.feature import hog
    arr = np.array(img.convert("RGB").resize((64, 64)))
    hog_feats = hog(arr, orientations=8, pixels_per_cell=(8,8),
                    cells_per_block=(2,2), channel_axis=-1)
    hist_r = np.histogram(arr[:,:,0], bins=16, range=(0,256))[0]
    hist_g = np.histogram(arr[:,:,1], bins=16, range=(0,256))[0]
    hist_b = np.histogram(arr[:,:,2], bins=16, range=(0,256))[0]
    colour_feats = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
    colour_feats /= colour_feats.sum() + 1e-6
    return np.concatenate([hog_feats, colour_feats]).reshape(1, -1)

def classify(img: Image.Image):
    bundle = load_model()
    model, scaler = bundle["model"], bundle["scaler"]
    feats = extract_features(img)
    feats_scaled = scaler.transform(feats)
    pred = model.predict(feats_scaled)[0]
    proba = model.predict_proba(feats_scaled)[0]
    confidence = proba.max()
    return pred, confidence

# ── WASTE INFO ──
WASTE = {
    "plastic": {
        "label": "Plastic",
        "emoji": "🧴",
        "color": "#DBEAFE",
        "accent": "#1D4ED8",
        "tag": "RECYCLABLE",
        "tag_color": "#1D4ED8",
        "en": "Place in the blue recycling bin. Rinse containers before disposal. Remove caps and lids where possible.",
        "hi": "नीले रीसाइकलिंग डब्बे में डालें। डालने से पहले कंटेनर अच्छे से धोएं।",
        "pa": "ਨੀਲੇ ਰੀਸਾਈਕਲਿੰਗ ਡੱਬੇ ਵਿੱਚ ਪਾਓ। ਪਾਉਣ ਤੋਂ ਪਹਿਲਾਂ ਡੱਬੇ ਧੋਵੋ।",
        "tip": "Single-use plastics take 400+ years to decompose. Switch to reusable alternatives.",
        "bin": "🔵 Blue Bin"
    },
    "organic": {
        "label": "Organic",
        "emoji": "🌿",
        "color": "#DCFCE7",
        "accent": "#15803D",
        "tag": "COMPOSTABLE",
        "tag_color": "#15803D",
        "en": "Place in the green compost bin. Can be used for home composting or as garden mulch.",
        "hi": "हरे खाद डब्बे में डालें। घर में खाद बनाने के लिए उपयोग करें।",
        "pa": "ਹਰੇ ਖਾਦ ਡੱਬੇ ਵਿੱਚ ਪਾਓ। ਘਰ ਵਿੱਚ ਖਾਦ ਬਣਾਉਣ ਲਈ ਵਰਤੋ।",
        "tip": "Composting organic waste reduces landfill volume by up to 30% and enriches soil.",
        "bin": "🟢 Green Bin"
    },
    "paper": {
        "label": "Paper",
        "emoji": "📦",
        "color": "#FEF9C3",
        "accent": "#A16207",
        "tag": "RECYCLABLE",
        "tag_color": "#A16207",
        "en": "Place in the recycling bin. Keep dry — wet paper cannot be recycled. Remove any plastic coating.",
        "hi": "रीसाइकलिंग डब्बे में डालें। सूखा रखें — गीला कागज रीसाइकल नहीं होता।",
        "pa": "ਰੀਸਾਈਕਲਿੰਗ ਡੱਬੇ ਵਿੱਚ ਪਾਓ। ਸੁੱਕਾ ਰੱਖੋ — ਗਿੱਲਾ ਕਾਗਜ਼ ਰੀਸਾਈਕਲ ਨਹੀਂ ਹੁੰਦਾ।",
        "tip": "Recycling one tonne of paper saves 17 trees and 26,000 litres of water.",
        "bin": "🟡 Yellow Bin"
    },
    "metal": {
        "label": "Metal",
        "emoji": "🥫",
        "color": "#F3E8FF",
        "accent": "#7E22CE",
        "tag": "RECYCLABLE",
        "tag_color": "#7E22CE",
        "en": "Place in the recycling bin. Metal is infinitely recyclable with no loss of quality.",
        "hi": "रीसाइकलिंग डब्बे में डालें। धातु को हमेशा रीसाइकल करें — गुणवत्ता नहीं घटती।",
        "pa": "ਰੀਸਾਈਕਲਿੰਗ ਡੱਬੇ ਵਿੱਚ ਪਾਓ। ਧਾਤ ਨੂੰ ਹਮੇਸ਼ਾ ਰੀਸਾਈਕਲ ਕਰੋ।",
        "tip": "Recycling aluminium uses 95% less energy than producing it from raw ore.",
        "bin": "🔵 Blue Bin"
    },
    "glass": {
        "label": "Glass",
        "emoji": "🫙",
        "color": "#CFFAFE",
        "accent": "#0E7490",
        "tag": "RECYCLABLE",
        "tag_color": "#0E7490",
        "en": "Place in the glass recycling bin. Wrap broken glass in newspaper before disposal for safety.",
        "hi": "कांच रीसाइकलिंग डब्बे में डालें। टूटे कांच को अखबार में लपेटकर फेंकें।",
        "pa": "ਸ਼ੀਸ਼ੇ ਦੇ ਰੀਸਾਈਕਲਿੰਗ ਡੱਬੇ ਵਿੱਚ ਪਾਓ। ਟੁੱਟੇ ਕੱਚ ਨੂੰ ਅਖਬਾਰ ਵਿੱਚ ਲਪੇਟੋ।",
        "tip": "Glass takes over 1 million years to decompose. Always recycle it.",
        "bin": "⚪ White Bin"
    },
    "ewaste": {
        "label": "E-Waste",
        "emoji": "💻",
        "color": "#FEE2E2",
        "accent": "#B91C1C",
        "tag": "HAZARDOUS",
        "tag_color": "#B91C1C",
        "en": "DO NOT place in regular bins. Take to nearest certified e-waste collection centre. Visit mcdonline.nic.in to find drop-off locations near you.",
        "hi": "सामान्य डब्बे में न डालें। निकटतम ई-अपशिष्ट संग्रह केंद्र पर ले जाएं।",
        "pa": "ਆਮ ਡੱਬੇ ਵਿੱਚ ਨਾ ਪਾਓ। ਨੇੜੇ ਦੇ ਈ-ਵੇਸਟ ਕੇਂਦਰ ਵਿੱਚ ਲੈ ਜਾਓ।",
        "tip": "E-waste contains lead, mercury and cadmium — toxic to soil and groundwater.",
        "bin": "🔴 E-Waste Centre"
    },
}

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

* { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #0F0F0F !important;
    color: #F0F0F0;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1100px; }

/* Nav bar */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 0 2rem;
    border-bottom: 1px solid #1E1E1E;
    margin-bottom: 2.5rem;
}
.nav-logo { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
.nav-logo span { color: #4ADE80; }
.nav-tag { font-size: 0.7rem; background: #1A1A1A; border: 1px solid #2E2E2E; color: #888; padding: 0.25rem 0.6rem; border-radius: 99px; }

/* Upload zone */
.upload-hint {
    text-align: center; padding: 3rem;
    border: 1px dashed #2E2E2E; border-radius: 16px;
    background: #111; margin: 1rem 0;
}
.upload-hint h3 { font-family: 'Syne', sans-serif; font-size: 1.1rem; color: #888; margin: 0.5rem 0; }

/* Result card */
.result-wrap {
    border-radius: 20px; overflow: hidden;
    border: 1px solid #2E2E2E; margin: 1.5rem 0;
}
.result-header {
    padding: 1.5rem 2rem;
    display: flex; align-items: center; gap: 1rem;
}
.result-emoji { font-size: 2.5rem; }
.result-label { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #fff; line-height: 1; }
.result-tag {
    display: inline-block; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 1.5px; padding: 0.2rem 0.7rem; border-radius: 4px;
    margin-top: 0.3rem;
}
.result-body { padding: 1.5rem 2rem; background: #111; }
.result-instruction { font-size: 1.05rem; line-height: 1.6; color: #E0E0E0; margin: 0 0 1rem; }
.result-bin { font-size: 0.9rem; font-weight: 600; color: #AAA; }
.result-tip { font-size: 0.85rem; color: #666; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #1E1E1E; }

/* Confidence bar */
.conf-wrap { margin: 1rem 0; }
.conf-label { font-size: 0.8rem; color: #666; margin-bottom: 0.4rem; display: flex; justify-content: space-between; }
.conf-track { height: 4px; background: #1E1E1E; border-radius: 99px; }
.conf-fill { height: 100%; border-radius: 99px; background: #4ADE80; }

/* Stats row */
.stats-row { display: flex; gap: 1rem; margin: 1.5rem 0; }
.stat-card {
    flex: 1; background: #111; border: 1px solid #1E1E1E;
    border-radius: 12px; padding: 1rem 1.2rem;
}
.stat-value { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #4ADE80; }
.stat-label { font-size: 0.75rem; color: #555; margin-top: 0.2rem; }

/* Category grid */
.cat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin: 1rem 0; }
.cat-card {
    background: #111; border: 1px solid #1E1E1E; border-radius: 12px;
    padding: 0.9rem 1rem; display: flex; align-items: center; gap: 0.7rem;
}
.cat-card-emoji { font-size: 1.3rem; }
.cat-card-label { font-size: 0.9rem; font-weight: 500; color: #CCC; }

/* Lang selector styling */
.stSelectbox > div > div { background: #111 !important; border: 1px solid #2E2E2E !important; color: #EEE !important; border-radius: 10px !important; }

/* File uploader */
.stFileUploader > div { background: #111 !important; border: 1px solid #2E2E2E !important; border-radius: 12px !important; }

/* Camera */
.stCameraInput > div { background: #111 !important; border: 1px solid #2E2E2E !important; border-radius: 12px !important; }

/* Divider */
hr { border-color: #1E1E1E !important; }

/* Section title */
.section-title { font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 700; color: #444; letter-spacing: 2px; text-transform: uppercase; margin: 2rem 0 1rem; }
</style>
""", unsafe_allow_html=True)

# ── NAVBAR ──
st.markdown("""
<div class="navbar">
    <div class="nav-logo">Vaste<span>AI</span></div>
    <div class="nav-tag">MINOR PROJECT · AMITY UNIVERSITY PUNJAB</div>
</div>
""", unsafe_allow_html=True)

# ── LAYOUT ──
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="section-title">Input</div>', unsafe_allow_html=True)

    lang = st.selectbox(
        "Language",
        options=["English", "हिंदी (Hindi)", "ਪੰਜਾਬੀ (Punjabi)"],
        label_visibility="collapsed"
    )
    lang_key = {"English": "en", "हिंदी (Hindi)": "hi", "ਪੰਜਾਬੀ (Punjabi)": "pa"}[lang]

    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
    camera   = st.camera_input("Camera", label_visibility="collapsed")
    image_source = camera if camera else uploaded

    if image_source:
        img = Image.open(image_source)
        st.image(img, use_container_width=True, clamp=True)
    else:
        st.markdown("""
        <div class="upload-hint">
            <div style="font-size:2.5rem">📸</div>
            <h3>Upload a photo or use camera</h3>
            <p style="color:#555; font-size:0.85rem; margin:0">Supports JPG, PNG, WEBP</p>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

    if image_source:
        with st.spinner(""):
            time.sleep(0.8)
            try:
                pred, confidence = classify(img)
            except Exception as e:
                st.error(f"Model error: {e}")
                st.stop()

        info = WASTE[pred]
        instruction = info[lang_key]

        # Result card
        st.markdown(f"""
        <div class="result-wrap">
            <div class="result-header" style="background: {info['color']}20;">
                <div class="result-emoji">{info['emoji']}</div>
                <div>
                    <div class="result-label">{info['label']}</div>
                    <div class="result-tag" style="background:{info['accent']}22; color:{info['accent']};">{info['tag']}</div>
                </div>
            </div>
            <div class="result-body">
                <p class="result-instruction">{instruction}</p>
                <div class="result-bin">{info['bin']}</div>
                <div class="conf-wrap">
                    <div class="conf-label">
                        <span>Confidence</span>
                        <span style="color:#4ADE80; font-weight:600">{confidence*100:.1f}%</span>
                    </div>
                    <div class="conf-track"><div class="conf-fill" style="width:{confidence*100:.1f}%"></div></div>
                </div>
                <div class="result-tip">💡 {info['tip']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Stats
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{info['label']}</div>
                <div class="stat-label">CATEGORY</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{confidence*100:.1f}%</div>
                <div class="stat-label">CONFIDENCE</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">SVM</div>
                <div class="stat-label">MODEL</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Audio
        st.markdown('<div class="section-title">Audio</div>', unsafe_allow_html=True)
        try:
            from gtts import gTTS
            gtts_lang = {"en": "en", "hi": "hi", "pa": "pa"}[lang_key]
            tts = gTTS(text=instruction, lang=gtts_lang, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            st.audio(buf, format="audio/mp3", autoplay=True)
        except Exception:
            st.caption("Install gTTS for audio playback")

    else:
        st.markdown('<div class="section-title">Categories</div>', unsafe_allow_html=True)
        st.markdown('<div class="cat-grid">', unsafe_allow_html=True)
        for key, info in WASTE.items():
            st.markdown(f"""
            <div class="cat-card">
                <span class="cat-card-emoji">{info['emoji']}</span>
                <span class="cat-card-label">{info['label']}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:2rem; padding:1.2rem; background:#111; border:1px solid #1E1E1E; border-radius:12px;">
            <div style="font-size:0.75rem; color:#444; letter-spacing:1.5px; font-weight:700; margin-bottom:0.8rem">MODEL INFO</div>
            <div style="font-size:0.9rem; color:#888; line-height:1.8">
                Algorithm: <span style="color:#CCC">SVM (RBF kernel)</span><br>
                Features: <span style="color:#CCC">HOG + Colour Histogram</span><br>
                Test Accuracy: <span style="color:#4ADE80; font-weight:600">99.6%</span><br>
                Classes: <span style="color:#CCC">6 waste categories</span><br>
                Dataset: <span style="color:#CCC">1,200 images</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("""
<div style="margin-top:3rem; padding-top:1.5rem; border-top:1px solid #1A1A1A;
     display:flex; justify-content:space-between; align-items:center;">
    <span style="font-size:0.8rem; color:#333">Ayachi Samyal · Pranav Bali</span>
    <span style="font-size:0.8rem; color:#333">Amity University Punjab · 2025–26</span>
</div>
""", unsafe_allow_html=True)
