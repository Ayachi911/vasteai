import streamlit as st
from PIL import Image
import numpy as np
import io, time, pickle, os

st.set_page_config(page_title=“VasteAI”, page_icon=“♻️”, layout=“wide”, initial_sidebar_state=“collapsed”)

@st.cache_resource
def load_model():
path = os.path.join(os.path.dirname(**file**), “model.pkl”)
with open(path, “rb”) as f:
return pickle.load(f)

def extract_features(img):
from skimage.feature import hog
arr = np.array(img.convert(‘RGB’).resize((64,64)))
hog_feats = hog(arr, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), channel_axis=-1)
hist_r = np.histogram(arr[:,:,0], bins=32, range=(0,256))[0]
hist_g = np.histogram(arr[:,:,1], bins=32, range=(0,256))[0]
hist_b = np.histogram(arr[:,:,2], bins=32, range=(0,256))[0]
colour = np.concatenate([hist_r,hist_g,hist_b]).astype(np.float32)
colour /= colour.sum()+1e-6
f = arr.astype(np.float32)/255.0
stats = np.array([f.mean(),f.std(),
f[:,:,0].mean()-f[:,:,2].mean(),f[:,:,1].mean()-f[:,:,0].mean(),
f[:,:,2].mean()-f[:,:,0].mean(),
1-(abs(f[:,:,0].mean()-f[:,:,1].mean())+abs(f[:,:,1].mean()-f[:,:,2].mean())),
(f>0.85).mean(),(f<0.2).mean(),f[:,:,0].std(),f[:,:,1].std(),f[:,:,2].std()])
return np.concatenate([hog_feats,colour,stats]).reshape(1,-1)

def classify(img):
b = load_model()
feats = b[‘scaler’].transform(extract_features(img))
pred = b[‘model’].predict(feats)[0]
conf = b[‘model’].predict_proba(feats)[0].max()
return pred, conf

WASTE = {
“glass”: {
“label”:“Glass”,“emoji”:“🍶”,“color”:”#ECFEFF”,“accent”:”#0E7490”,“tag”:“RECYCLABLE”,
“en”:“Place in the glass recycling bin. Wrap broken glass in newspaper before disposal.”,
“hi”:“कांच रीसाइकलिंग डब्बे में डालें। टूटे कांच को अखबार में लपेटें।”,
“pa”:“ਸ਼ੀਸ਼ੇ ਦੇ ਰੀਸਾਈਕਲਿੰਗ ਡੱਬੇ ਵਿੱਚ ਪਾਓ। ਟੁੱਟੇ ਕੱਚ ਨੂੰ ਅਖਬਾਰ ਵਿੱਚ ਲਪੇਟੋ।”,
“tip”:“Glass takes over 1 million years to decompose in landfill. Always recycle it.”,“bin”:“⚪ White Bin”
},
“metal”: {
“label”:“Metal”,“emoji”:“🔩”,“color”:”#FAF5FF”,“accent”:”#7E22CE”,“tag”:“RECYCLABLE”,
“en”:“Place in the recycling bin. Metal is infinitely recyclable with no quality loss.”,
“hi”:“रीसाइकलिंग डब्बे में डालें। धातु को हमेशा रीसाइकल करें।”,
“pa”:“ਰੀਸਾਈਕਲਿੰਗ ਡੱਬੇ ਵਿੱਚ ਪਾਓ। ਧਾਤ ਨੂੰ ਹਮੇਸ਼ਾ ਰੀਸਾਈਕਲ ਕਰੋ।”,
“tip”:“Recycling aluminium uses 95% less energy than producing it from raw ore.”,“bin”:“🔵 Blue Bin”
},
“plastic”: {
“label”:“Plastic”,“emoji”:“♻️”,“color”:”#EFF6FF”,“accent”:”#1D4ED8”,“tag”:“RECYCLABLE”,
“en”:“Place in the blue recycling bin. Rinse containers before disposal. Remove caps if possible.”,
“hi”:“नीले रीसाइकलिंग डब्बे में डालें। फेंकने से पहले कंटेनर धोएं।”,
“pa”:“ਨੀਲੇ ਰੀਸਾਈਕਲਿੰਗ ਡੱਬੇ ਵਿੱਚ ਪਾਓ। ਸੁੱਟਣ ਤੋਂ ਪਹਿਲਾਂ ਡੱਬੇ ਧੋਵੋ।”,
“tip”:“Single-use plastics take 400+ years to decompose. Switch to reusable alternatives.”,“bin”:“🔵 Blue Bin”
},
“trash”: {
“label”:“General Waste”,“emoji”:“🗑️”,“color”:”#F1F5F9”,“accent”:”#475569”,“tag”:“LANDFILL”,
“en”:“Place in the general waste bin. Cannot be recycled — dispose responsibly.”,
“hi”:“सामान्य कचरे के डब्बे में डालें। इसे रीसाइकल नहीं किया जा सकता।”,
“pa”:“ਆਮ ਕੂੜੇ ਦੇ ਡੱਬੇ ਵਿੱਚ ਪਾਓ। ਇਸਨੂੰ ਰੀਸਾਈਕਲ ਨਹੀਂ ਕੀਤਾ ਜਾ ਸਕਦਾ।”,
“tip”:“Reduce general waste by choosing products with less packaging.”,“bin”:“🖤 Black Bin”
},
}

st.markdown(”””

<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');
* { box-sizing: border-box; }
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; background-color: #0F0F0F !important; color: #F0F0F0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1100px; }
.navbar { display:flex; align-items:center; justify-content:space-between; padding:1rem 0 2rem; border-bottom:1px solid #1E1E1E; margin-bottom:2.5rem; }
.nav-logo { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; color:#fff; letter-spacing:-0.5px; }
.nav-logo span { color:#4ADE80; }
.nav-tag { font-size:0.7rem; background:#1A1A1A; border:1px solid #2E2E2E; color:#888; padding:0.25rem 0.6rem; border-radius:99px; }
.result-wrap { border-radius:20px; overflow:hidden; border:1px solid #2E2E2E; margin:1.5rem 0; }
.result-header { padding:1.5rem 2rem; display:flex; align-items:center; gap:1rem; }
.result-emoji { font-size:2.5rem; }
.result-label { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#fff; line-height:1; }
.result-tag { display:inline-block; font-size:0.65rem; font-weight:700; letter-spacing:1.5px; padding:0.2rem 0.7rem; border-radius:4px; margin-top:0.3rem; }
.result-body { padding:1.5rem 2rem; background:#111; }
.result-instruction { font-size:1.05rem; line-height:1.6; color:#E0E0E0; margin:0 0 1rem; }
.result-bin { font-size:0.9rem; font-weight:600; color:#AAA; }
.result-tip { font-size:0.85rem; color:#666; margin-top:1rem; padding-top:1rem; border-top:1px solid #1E1E1E; }
.conf-wrap { margin:1rem 0; }
.conf-label { font-size:0.8rem; color:#666; margin-bottom:0.4rem; display:flex; justify-content:space-between; }
.conf-track { height:4px; background:#1E1E1E; border-radius:99px; }
.conf-fill { height:100%; border-radius:99px; background:#4ADE80; }
.stats-row { display:flex; gap:1rem; margin:1.5rem 0; }
.stat-card { flex:1; background:#111; border:1px solid #1E1E1E; border-radius:12px; padding:1rem 1.2rem; }
.stat-value { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#4ADE80; }
.stat-label { font-size:0.75rem; color:#555; margin-top:0.2rem; }
.cat-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:0.75rem; margin:1rem 0; }
.cat-card { background:#111; border:1px solid #1E1E1E; border-radius:12px; padding:0.9rem 1rem; display:flex; align-items:center; gap:0.7rem; }
.stSelectbox > div > div { background:#111 !important; border:1px solid #2E2E2E !important; color:#EEE !important; border-radius:10px !important; }
.section-title { font-family:'Syne',sans-serif; font-size:0.75rem; font-weight:700; color:#444; letter-spacing:2px; text-transform:uppercase; margin:2rem 0 1rem; }
</style>

“””, unsafe_allow_html=True)

st.markdown(”””

<div class="navbar">
    <div class="nav-logo">Vaste<span>AI</span></div>
    <div class="nav-tag">MINOR PROJECT · AMITY UNIVERSITY PUNJAB</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.1, 0.9], gap=“large”)

with left:
st.markdown(’<div class="section-title">Input</div>’, unsafe_allow_html=True)
lang = st.selectbox(“Language”, options=[“English”,“हिंदी (Hindi)”,“ਪੰਜਾਬੀ (Punjabi)”], label_visibility=“collapsed”)
lang_key = {“English”:“en”,“हिंदी (Hindi)”:“hi”,“ਪੰਜਾਬੀ (Punjabi)”:“pa”}[lang]
uploaded = st.file_uploader(“Upload image”, type=[“jpg”,“jpeg”,“png”,“webp”], label_visibility=“collapsed”)
camera   = st.camera_input(“Camera”, label_visibility=“collapsed”)
image_source = camera if camera else uploaded
if image_source:
img = Image.open(image_source)
st.image(img, use_container_width=True)
else:
st.markdown(’<div style="text-align:center;padding:3rem;border:1px dashed #2E2E2E;border-radius:16px;background:#111;"><div style="font-size:2.5rem">📸</div><p style="color:#555;margin:0.5rem 0 0">Upload a photo or use camera</p></div>’, unsafe_allow_html=True)

with right:
st.markdown(’<div class="section-title">Result</div>’, unsafe_allow_html=True)
if image_source:
with st.spinner(””):
time.sleep(0.8)
pred, confidence = classify(img)

```
    info = WASTE[pred]
    instruction = info[lang_key]
    lang_labels = {"en":"English","hi":"हिंदी","pa":"ਪੰਜਾਬੀ"}

    st.markdown(f"""
    <div class="result-wrap">
        <div class="result-header" style="background:{info['color']}20;">
            <div class="result-emoji">{info['emoji']}</div>
            <div>
                <div class="result-label">{info['label']}</div>
                <div class="result-tag" style="background:{info['accent']}22;color:{info['accent']};">{info['tag']}</div>
            </div>
        </div>
        <div class="result-body">
            <p class="result-instruction">{instruction}</p>
            <div class="result-bin">{info['bin']}</div>
            <div class="conf-wrap">
                <div class="conf-label"><span>Confidence</span><span style="color:#4ADE80;font-weight:600">{confidence*100:.1f}%</span></div>
                <div class="conf-track"><div class="conf-fill" style="width:{confidence*100:.1f}%"></div></div>
            </div>
            <div class="result-tip">💡 {info['tip']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card"><div class="stat-value">{info['label']}</div><div class="stat-label">CATEGORY</div></div>
        <div class="stat-card"><div class="stat-value">{confidence*100:.1f}%</div><div class="stat-label">CONFIDENCE</div></div>
        <div class="stat-card"><div class="stat-value">RF</div><div class="stat-label">MODEL</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Audio</div>', unsafe_allow_html=True)
    try:
        from gtts import gTTS
        tts = gTTS(text=instruction, lang={"en":"en","hi":"hi","pa":"pa"}[lang_key], slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf); buf.seek(0)
        st.audio(buf, format="audio/mp3", autoplay=True)
    except:
        st.caption("Install gTTS for audio")
else:
    st.markdown('<div class="section-title">Categories</div>', unsafe_allow_html=True)
    st.markdown('<div class="cat-grid">', unsafe_allow_html=True)
    for k,v in WASTE.items():
        st.markdown(f'<div class="cat-card"><span style="font-size:1.3rem">{v["emoji"]}</span><span style="font-size:0.9rem;font-weight:500;color:#CCC">{v["label"]}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top:2rem;padding:1.2rem;background:#111;border:1px solid #1E1E1E;border-radius:12px;">
        <div style="font-size:0.75rem;color:#444;letter-spacing:1.5px;font-weight:700;margin-bottom:0.8rem">MODEL INFO</div>
        <div style="font-size:0.9rem;color:#888;line-height:1.8">
            Algorithm: <span style="color:#CCC">Random Forest (100 trees)</span><br>
            Features: <span style="color:#CCC">HOG + Colour Histogram</span><br>
            Test Accuracy: <span style="color:#4ADE80;font-weight:600">69.3%</span><br>
            Dataset: <span style="color:#CCC">TrashNet — 1,530 real images</span><br>
            Classes: <span style="color:#CCC">Glass, Metal, Plastic, Trash</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
```

st.markdown(’<div style="margin-top:3rem;padding-top:1.5rem;border-top:1px solid #1A1A1A;display:flex;justify-content:space-between;"><span style="font-size:0.8rem;color:#333">Ayachi Samyal · Pranav Bali</span><span style="font-size:0.8rem;color:#333">Amity University Punjab · 2025–26</span></div>’, unsafe_allow_html=True)
