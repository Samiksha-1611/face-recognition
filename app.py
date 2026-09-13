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
    .stButton>button:hover {
        border-color: #00ffc8; box-shadow: 0 0 14px rgba(0,255,200,0.4); color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Shared state -----------------
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

tab_identify, tab_enroll, tab_db, tab_eval = st.tabs(["Identify", "Enroll", "Database", "Evaluation"])

# ----------------- Identify -----------------
with tab_identify:
    st.markdown("Upload an image containing a face to identify the person against the enrolled database.")

    threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=DEFAULT_THRESHOLD, step=0.01)
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="identify_upload")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        col_img, col_result = st.columns([1, 1.4])
        with col_img:
            st.image(image, caption="Uploaded Image", width='stretch')

        with col_result:
            if st.button("Run Identification", key="run_id"):
                with st.spinner("Analyzing..."):
                    img_array = np.array(image)
                    img_bgr = img_array[..., ::-1]  # RGB to BGR
                    result = identify_face(img_bgr, threshold=threshold)

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
                            <p class='verdict-unknown'>UNKNOWN PERSON</p>
                            <p><b>Best Candidate:</b> {result.get('identity', 'N/A')}</p>
                            <p><b>Similarity:</b> {result['similarity']:.4f}</p>
                            <p><b>Threshold:</b> {result['threshold']:.4f}</p>
                            <p><b>Decision:</b> REJECTED</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Error: {result.get('message', 'Unknown error occurred.')}")

# ----------------- Enroll -----------------
with tab_enroll:
    st.markdown("Upload 2–3 clear images of the same person to add them to the system.")

    person_name = st.text_input("Person Name")
    uploaded_files = st.file_uploader(
        "Upload images for enrollment", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, key="enroll_upload"
    )

    if st.button("Enroll Person", key="enroll_btn"):
        if not person_name:
            st.error("Please enter a name.")
        elif not uploaded_files:
            st.error("Please upload at least one image.")
        else:
            success_count = 0
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
                st.success(f"Enrolled {success_count} image(s) for {person_name}")
                refresh_stats()
                st.rerun()

# ----------------- Database -----------------
with tab_db:
    stats = st.session_state.db_stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='stat-card'><h3>{stats['num_people']}</h3><p>ENROLLED PEOPLE</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-card'><h3>{stats['num_embeddings']}</h3><p>TOTAL EMBEDDINGS</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    db = load_database()
    if not db:
        st.info("Database is empty.")
    else:
        table_data = [
            {"Name": name, "Number of Images": len(data.get("embeddings", []))}
            for name, data in db.items()
        ]
        st.dataframe(pd.DataFrame(table_data), width='stretch')

# ----------------- Evaluation -----------------
with tab_eval:
    st.info(f"**Current Application Threshold:** {DEFAULT_THRESHOLD}  \n"
            f"Calibrated using enrollment and out-of-sample test images.")

    st.markdown("---")
    st.subheader("External Evaluation (LFW)")

    metrics_file = os.path.join(EVAL_DIR, "metrics.json")
    plot_file = os.path.join(EVAL_DIR, "similarity_distribution.png")

    if os.path.exists(metrics_file) and os.path.exists(plot_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='stat-card'><h3>{metrics['total_pairs_evaluated']}</h3><p>PAIRS EVALUATED</p></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat-card'><h3>{metrics['best_threshold']:.2f}</h3><p>RECOMMENDED THRESHOLD</p></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='stat-card'><h3>{metrics['best_f1']:.3f}</h3><p>BEST F1 SCORE</p></div>", unsafe_allow_html=True)

        st.markdown("###")
        st.image(plot_file, caption="Genuine vs Impostor Cosine Similarity Distribution")

        st.subheader("Metrics by Threshold")
        df_metrics = pd.DataFrame(metrics["metrics_by_threshold"])
        st.dataframe(df_metrics.style.highlight_max(subset=['f1'], color='#0d3d33'), width='stretch')
    else:
        st.warning("No external evaluation results found.")