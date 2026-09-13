import streamlit as st
import pandas as pd
import json
import os
from PIL import Image
import numpy as np

# Set up page configuration first
st.set_page_config(page_title="Face Recognition ID System", layout="wide", initial_sidebar_state="collapsed")

from src.enroll import process_and_enroll_image
from src.identify import identify_face
from src.database import get_database_stats, load_database
from src.config import MODEL_NAME, DETECTOR_BACKEND, DEFAULT_THRESHOLD, EVAL_DIR

# ----------------- Cyber Theme CSS -----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Orbitron:wght@600;800&display=swap');

    .stApp {
        background: radial-gradient(circle at 20% 0%, #0a1418 0%, #060a0d 55%, #04070a 100%);
        color: #d6f5ec;
    }
    * { font-family: 'JetBrains Mono', monospace; }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; letter-spacing: 0.5px; }

    /* Persistent status header */
    .status-bar {
        display: flex; justify-content: space-between; align-items: center;
        background: linear-gradient(90deg, rgba(0,255,200,0.06), rgba(0,150,255,0.06));
        border: 1px solid rgba(0,255,200,0.25);
        border-radius: 10px; padding: 14px 22px; margin-bottom: 22px;
        box-shadow: 0 0 18px rgba(0,255,200,0.08);
    }
    .status-item { font-size: 13px; color: #7ff4d6; }
    .status-item b { color: #ffffff; }

    /* Tabs styled as button sections */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: none; margin-bottom: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(0,255,200,0.04); border: 1px solid rgba(0,255,200,0.25);
        border-radius: 8px; color: #7ff4d6; padding: 12px 22px; font-weight: 600;
        transition: 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0,255,200,0.1); border-color: rgba(0,255,200,0.5);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0,255,200,0.18) !important; color: #ffffff !important;
        border: 1px solid #00ffc8 !important;
        box-shadow: 0 0 14px rgba(0,255,200,0.4);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* Cards / panels */
    .stat-card {
        background: rgba(0,255,200,0.04); border: 1px solid rgba(0,255,200,0.25);
        border-radius: 12px; padding: 22px; text-align: center;
        box-shadow: 0 0 14px rgba(0,255,200,0.08);
    }
    .stat-card h3 { font-size: 30px; margin: 0; color: #00ffc8; text-shadow: 0 0 10px rgba(0,255,200,0.6); }
    .stat-card p { color: #9adfd0; margin: 4px 0 0 0; font-size: 13px; }

    .result-matched {
        border: 1px solid rgba(0,255,150,0.5); background: rgba(0,255,150,0.06);
        border-radius: 12px; padding: 22px; margin-top: 18px;
        box-shadow: 0 0 22px rgba(0,255,150,0.25);
    }
    .result-unknown {
        border: 1px solid rgba(255,80,80,0.5); background: rgba(255,80,80,0.06);
        border-radius: 12px; padding: 22px; margin-top: 18px;
        box-shadow: 0 0 22px rgba(255,80,80,0.2);
    }
    .verdict-matched { color: #00ffb0; font-size: 22px; font-weight: 700; text-shadow: 0 0 8px rgba(0,255,150,0.6); }
    .verdict-unknown { color: #ff5c5c; font-size: 22px; font-weight: 700; text-shadow: 0 0 8px rgba(255,80,80,0.5); }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, rgba(0,255,200,0.15), rgba(0,150,255,0.15));
        border: 1px solid rgba(0,255,200,0.4); color: #d6f5ec; border-radius: 8px;
        transition: 0.2s;
    }
    /* Metric info box (replaces default st.info blue) */
    .metric-info-box {
        border: 1px solid rgba(0,255,200,0.25); background: rgba(0,255,200,0.05);
        border-radius: 10px; padding: 18px 22px; color: #d6f5ec; font-size: 14px;
        line-height: 1.6; margin-bottom: 4px;
    }
    .metric-info-box b { color: #00ffc8; }

    /* Dark-themed dataframe/table container (border only — do not touch internals,
       Streamlit's dataframe renders on an internal canvas that CSS color overrides break) */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(0,255,200,0.2) !important;
        border-radius: 10px !important;
    }

    /* Intro feature cards (rendered as buttons) */
    .intro-subtitle {
        color: #9adfd0; font-size: 14px; line-height: 1.6; margin: 6px 0 22px 0; max-width: 780px;
    }
    div[data-testid="column"] .stButton>button {
        height: 64px; font-size: 15px; font-weight: 700;
        background: rgba(0,255,200,0.03); border: 1px solid rgba(0,255,200,0.2);
    }
    .feature-desc { color: #8fa89f; font-size: 12px; line-height: 1.4; }

    /* Biometric Scanner Animation */
    .scanner-container {
        position: relative;
        display: inline-block;
        overflow: hidden;
        border: 1px solid rgba(0,255,200,0.5);
        border-radius: 8px;
        box-shadow: 0 0 15px rgba(0,255,200,0.15);
        width: 100%;
    }
    .scanner-container img {
        display: block;
        width: 100%;
        height: auto;
    }
    .scanner-line {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: #00ffc8;
        box-shadow: 0 0 10px #00ffc8, 0 0 20px #00ffc8;
        animation: scan 2s infinite linear;
    }
    .scanner-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(to bottom, rgba(0,255,200,0) 0%, rgba(0,255,200,0.1) 50%, rgba(0,255,200,0) 100%);
        animation: scan-glow 2s infinite linear;
    }
    @keyframes scan {
        0% { top: -5%; }
        50% { top: 105%; }
        100% { top: -5%; }
    }
    @keyframes scan-glow {
        0% { top: -50%; }
        50% { top: 50%; }
        100% { top: -50%; }
    }
    .scanner-label {
        position: absolute;
        bottom: 10px;
        left: 10px;
        color: #00ffc8;
        font-size: 13px;
        font-weight: bold;
        text-shadow: 0 0 5px #00ffc8;
        background: rgba(0,0,0,0.6);
        padding: 4px 8px;
        border-radius: 4px;
        animation: pulse 1s infinite alternate;
    }
    @keyframes pulse {
        0% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .system-status-list {
        font-family: 'JetBrains Mono', monospace;
        color: #8fa89f;
        font-size: 14px;
        line-height: 1.8;
    }
    .system-status-list b { color: #00ffc8; }
    
    /* Camera constraint */
    [data-testid="stCameraInput"] {
        max-width: 500px !important;
        margin: 0 auto;
        border: 1px solid rgba(0,255,200,0.4);
        border-radius: 8px;
        padding: 5px;
        background: rgba(0,255,200,0.03);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Shared state -----------------
import base64
from io import BytesIO

def get_image_base64(img: Image.Image):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

if "db_stats" not in st.session_state:
    st.session_state.db_stats = get_database_stats()

def refresh_stats():
    st.session_state.db_stats = get_database_stats()

@st.cache_resource
def init_models():
    pass

init_models()

# ----------------- Persistent status header -----------------
stats = st.session_state.db_stats
st.markdown(f"""
<div class="status-bar">
    <div class="status-item"><b>{stats['num_people']}</b> people enrolled &nbsp;|&nbsp; <b>{stats['num_embeddings']}</b> embeddings stored</div>
    <div class="status-item">Model: <b>{MODEL_NAME}</b> &nbsp;|&nbsp; Detector: <b>{DETECTOR_BACKEND}</b> &nbsp;|&nbsp; Threshold: <b>{DEFAULT_THRESHOLD}</b></div>
</div>
""", unsafe_allow_html=True)

st.markdown("# FACE RECOGNITION IDENTIFICATION SYSTEM")
st.markdown("""
<p class="intro-subtitle">
Enroll identities and identify faces using AI-powered biometric matching.
</p>
""", unsafe_allow_html=True)

if "active_section" not in st.session_state:
    st.session_state.active_section = "Identify"

intro_cols = st.columns(3)
cards = [
    ("ENROLL PERSON", "Register a new identity.", "material/person_add", "Enroll"),
    ("IDENTIFY FACE", "Check a photo against everyone enrolled.", "material/search", "Identify"),
    ("DATABASE", "View enrolled identities.", "material/grid_view", "Database"),
]
for col, (title, desc, icon, section) in zip(intro_cols, cards):
    with col:
        if st.button(title, key=f"card_{section}", icon=f":{icon}:", width="stretch"):
            st.session_state.active_section = section
        st.markdown(f'<p class="feature-desc" style="text-align:center; margin-top:-6px;">{desc}</p>', unsafe_allow_html=True)

active = st.session_state.active_section
st.markdown("###")

# ----------------- Identify -----------------
if active == "Identify":
    st.markdown("### FACE IDENTIFIER")
    st.markdown("────────────────────────")

    threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=DEFAULT_THRESHOLD, step=0.01)
    
    input_type = st.radio("Select Input Method", ["UPLOAD IMAGE", "LIVE CAMERA"], horizontal=True, label_visibility="collapsed")
    
    uploaded_file = None
    if input_type == "UPLOAD IMAGE":
        st.markdown("<p style='font-size:14px; color:#8fa89f;'>Upload a photo for identification</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="identify_upload", label_visibility="collapsed")
    else:
        st.markdown("<p style='font-size:14px; color:#8fa89f;'>Scan a face using your webcam</p>", unsafe_allow_html=True)
        uploaded_file = st.camera_input("Take a picture", label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        col_img, col_result = st.columns([1, 1.4])
        
        with col_img:
            img_container = st.empty()
            img_container.image(image, caption="Uploaded Image", width='stretch')

        with col_result:
            st.markdown("### SYSTEM STATUS")
            st.markdown("""<div class="system-status-list">
                ● FACE DETECTION &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>READY</b><br>
                ● EMBEDDING ENGINE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>READY</b><br>
                ● DATABASE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>READY</b><br>
                ● IDENTIFICATION &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>READY</b>
            </div>""", unsafe_allow_html=True)
            
            st.markdown("###")
            if st.button("SCAN / ANALYZE FACE", key="run_id", use_container_width=True):
                # Replace image with scanner
                b64 = get_image_base64(image)
                img_container.markdown(f'''
                <div class="scanner-container">
                    <img src="data:image/jpeg;base64,{b64}" />
                    <div class="scanner-line"></div>
                    <div class="scanner-overlay"></div>
                    <div class="scanner-label">SCANNING...</div>
                </div>
                ''', unsafe_allow_html=True)
                
                with st.spinner("Analyzing biometric markers..."):
                    img_array = np.array(image)
                    img_bgr = img_array[..., ::-1]  # RGB to BGR
                    result = identify_face(img_bgr, threshold=threshold)
                    
                    # Restore image
                    img_container.empty()
                    img_container.image(image, caption="Analyzed Image", width='stretch')
                    
                    if result["status"] == "matched":
                        st.markdown(f"""
                        <div class="result-matched">
                            <p class='verdict-matched'>MATCH FOUND</p>
                            <p><b>Identity:</b> {result['identity']}</p>
                            <p><b>Similarity:</b> {result['similarity']:.4f}</p>
                            <p><b>Threshold:</b> {result['threshold']:.4f}</p>
                            <p><b>Decision:</b> ACCEPTED</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif result["status"] == "unknown":
                        st.markdown(f"""
                        <div class="result-unknown">
                            <p class='verdict-unknown'>UNKNOWN FACE</p>
                            <p style='margin-bottom:12px;'>No enrolled identity matched this face.</p>
                            <p><b>Similarity:</b> {result['similarity']:.4f}</p>
                            <p><b>Threshold:</b> {result['threshold']:.4f}</p>
                            <p><b>Decision:</b> REJECTED</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-unknown">
                            <p class='verdict-unknown'>FACE NOT DETECTED</p>
                            <p>Please capture or upload an image containing a clear face.</p>
                            <p>Error: {result.get('message', 'Unknown error')}</p>
                        </div>
                        """, unsafe_allow_html=True)

# ----------------- Enroll -----------------
elif active == "Enroll":
    st.markdown("### ENROLL NEW IDENTITY")
    st.markdown("────────────────────────")

    person_name = st.text_input("IDENTITY NAME")
    uploaded_files = st.file_uploader(
        "FACE SAMPLES (Upload 2–3 images)", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, key="enroll_upload"
    )
    
    st.markdown("### SYSTEM CHECK")
    st.markdown("""<div class="system-status-list">
        ● Face detection &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>READY</b><br>
        ● Embedding generation &nbsp; <b>READY</b><br>
        ● Database storage &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>READY</b>
    </div>""", unsafe_allow_html=True)
    st.markdown("###")

    if st.button("ENROLL PERSON", key="enroll_btn", use_container_width=True):
        if not person_name:
            st.error("Please enter a name.")
        elif not uploaded_files:
            st.error("Please upload at least one image.")
        else:
            success_count = 0
            with st.spinner("Processing face embeddings..."):
                for file in uploaded_files:
                    image = Image.open(file).convert('RGB')
                    img_array = np.array(image)
                    img_bgr = img_array[..., ::-1]
    
                    result = process_and_enroll_image(person_name, img_bgr)
                    if result["success"]:
                        success_count += 1
                    else:
                        st.error(f"{file.name} — Rejected: {result['message']}")

            if success_count > 0:
                refresh_stats()
                st.markdown(f"""
                <div class="result-matched">
                    <p class='verdict-matched'>IDENTITY REGISTERED</p>
                    <p><b>Name:</b> {person_name}</p>
                    <p><b>Images processed:</b> {success_count}</p>
                    <p><b>Embeddings stored:</b> {success_count}</p>
                </div>
                """, unsafe_allow_html=True)

# ----------------- Database -----------------
elif active == "Database":
    st.markdown("### ENROLLED IDENTITIES")
    st.markdown("────────────────────────")
    
    stats = st.session_state.db_stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='stat-card'><h3>{stats['num_people']}</h3><p>ENROLLED PEOPLE</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-card'><h3>{stats['num_embeddings']}</h3><p>TOTAL EMBEDDINGS</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    db = load_database()
    if not db:
        st.markdown('<div class="metric-info-box">Database is empty.</div>', unsafe_allow_html=True)
    else:
        table_data = [
            {"Name": name, "Number of Images": len(data.get("embeddings", []))}
            for name, data in db.items()
        ]
        st.dataframe(pd.DataFrame(table_data), width='stretch')
