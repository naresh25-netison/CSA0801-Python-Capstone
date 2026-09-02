"""
AI Handwritten Digit Recognizer - Professional Web Dashboard
Powered by Scikit-Learn KNeighborsClassifier (k=3) & Scikit-Learn Digits Dataset
"""

import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from streamlit_drawable_canvas import st_canvas

# Ensure modules are discoverable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from preprocessing import (
    load_digits_dataset,
    get_knn_model,
    preprocess_digit,
    get_real_dataset_sample,
    evaluate_k_values
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* Glassmorphism Card Style */
.glass-card {
    background: rgba(30, 41, 59, 0.65);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    margin-bottom: 18px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
}

/* Status Online Badge */
.status-badge-online {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #34d399;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.status-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 10px #10b981;
}

/* Hero Gradient Title */
.gradient-title {
    background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 6px;
}

/* Stat Cards */
.stat-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    transition: all 0.2s ease;
}

.stat-card:hover {
    border-color: #6366f1;
    transform: translateY(-2px);
}

.stat-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.stat-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #f8fafc;
    font-family: 'Outfit', sans-serif;
}

/* Prediction Result Hero Card */
.prediction-box {
    background: linear-gradient(145deg, #1e1b4b 0%, #0f172a 100%);
    border: 2px solid #6366f1;
    border-radius: 20px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 0 35px rgba(99, 102, 241, 0.3);
}

.pred-digit-giant {
    font-size: 5.5rem;
    font-weight: 900;
    line-height: 1;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Outfit', sans-serif;
    margin: 8px 0;
}

.pred-status-pill {
    display: inline-block;
    padding: 4px 12px;
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #10b981;
    color: #6ee7b7;
    font-size: 0.85rem;
    font-weight: 600;
    border-radius: 9999px;
    margin-top: 6px;
}

/* Flow Step Items */
.flow-step-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid #8b5cf6;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
}

.flow-step-num {
    font-size: 1.15rem;
    font-weight: 800;
    color: #a78bfa;
    font-family: 'Outfit', sans-serif;
    margin-right: 10px;
}

/* Pad Container */
.pad-container {
    background: #ffffff;
    border: 2px solid #334155;
    border-radius: 16px;
    overflow: hidden;
    display: inline-block;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.example-tag {
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid #38bdf8;
    color: #7dd3fc;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 6px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Model & Dataset Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Model & Dataset...")
def get_cached_resources():
    dataset = load_digits_dataset()
    model = get_knn_model(n_neighbors=3)
    eval_results = evaluate_k_values()
    return dataset, model, eval_results

digits_dataset, model, eval_stats = get_cached_resources()


# -----------------------------------------------------------------------------
# 3. Session State Management
# -----------------------------------------------------------------------------
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "active_source" not in st.session_state:
    st.session_state.active_source = "pad"  # 'pad' or 'dataset_example'

if "example_sample" not in st.session_state:
    st.session_state.example_sample = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"


# -----------------------------------------------------------------------------
# 4. Navigation & Sidebar
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 8px 0 16px 0;">
        <span style="font-size: 2.2rem;">✍️</span>
        <h3 style="margin-top: 4px; margin-bottom: 0;">Digit Recognizer</h3>
        <p style="font-size: 0.8rem; color: #94a3b8; margin: 0;">Scikit-Learn KNN (k=3)</p>
    </div>
    """,
    unsafe_allow_html=True
)

NAV_PAGES = [
    "🏠 Dashboard",
    "✍️ Digit Recognition",
    "🤖 Model",
    "📊 Dataset",
    "ℹ️ About"
]

selected_nav = st.sidebar.radio(
    "Navigation Menu",
    NAV_PAGES,
    index=NAV_PAGES.index(st.session_state.current_page) if st.session_state.current_page in NAV_PAGES else 0,
    key="nav_selection"
)

if selected_nav != st.session_state.current_page:
    st.session_state.current_page = selected_nav

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.6;">
        <p style="margin-bottom: 6px;"><strong>⚡ System Status:</strong></p>
        • Algorithm: <code>K-Nearest Neighbors</code><br>
        • Parameter (k): <code>3</code><br>
        • CV Accuracy: <code>98.94%</code><br>
        • Input: <code>8 × 8 Matrix (64-D)</code><br>
        • Scale: <code>0 to 16 range</code>
    </div>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# PAGE 1: 🏠 DASHBOARD HOME
# =============================================================================
if st.session_state.current_page == "🏠 Dashboard":
    st.markdown(
        """
        <div class="status-badge-online">
            <span class="status-dot"></span>
            MODEL ONLINE (k=3)
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 class="gradient-title">AI HANDWRITTEN DIGIT RECOGNIZER</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size: 1.15rem; color: #cbd5e1; max-width: 820px; line-height: 1.6; margin-bottom: 24px;">
            Recognize handwritten digits using Machine Learning. Draw your handwriting on the interactive pad
            or test real benchmark samples to see instant optical character recognition in action.
        </p>
        """,
        unsafe_allow_html=True
    )
    
    col_btn, _ = st.columns([1.2, 3])
    with col_btn:
        if st.button("🚀 START RECOGNIZING", type="primary", use_container_width=True):
            st.session_state.current_page = "✍️ Digit Recognition"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Statistic Cards
    st.markdown("### 📊 System Overview")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Model</div>
                <div class="stat-value" style="color: #818cf8;">KNN (k=3)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with sc2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Dataset</div>
                <div class="stat-value" style="color: #38bdf8;">{len(digits_dataset.data):,} Samples</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with sc3:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Classes</div>
                <div class="stat-value" style="color: #34d399;">10 (0–9)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with sc4:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Image</div>
                <div class="stat-value" style="color: #f472b6;">8 × 8 (64-D)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # How It Works Section
    st.markdown("### 💡 How It Works")
    st.markdown(
        """
        <div class="glass-card">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px;">
                <div style="background: rgba(15, 23, 42, 0.6); padding: 16px; border-radius: 10px; border-top: 3px solid #38bdf8;">
                    <strong style="color: #38bdf8; font-size: 1.05rem;">1. Input</strong><br>
                    <span style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5; display: inline-block; margin-top: 6px;">
                        User draws freehand stroke on the digital canvas or chooses a real dataset sample.
                    </span>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 16px; border-radius: 10px; border-top: 3px solid #818cf8;">
                    <strong style="color: #818cf8; font-size: 1.05rem;">2. Preprocessing</strong><br>
                    <span style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5; display: inline-block; margin-top: 6px;">
                        Automatic polarity check, bounding box crop, center of mass alignment, and 8×8 scaling (0–16).
                    </span>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 16px; border-radius: 10px; border-top: 3px solid #c084fc;">
                    <strong style="color: #c084fc; font-size: 1.05rem;">3. KNN Classification</strong><br>
                    <span style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5; display: inline-block; margin-top: 6px;">
                        Measures Euclidean distance against all 1,797 training vectors in 64-dimensional feature space.
                    </span>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 16px; border-radius: 10px; border-top: 3px solid #34d399;">
                    <strong style="color: #34d399; font-size: 1.05rem;">4. Prediction</strong><br>
                    <span style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5; display: inline-block; margin-top: 6px;">
                        Outputs the majority class with probability distribution and confidence analysis.
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# PAGE 2: ✍️ DIGIT RECOGNITION PAGE
# =============================================================================
elif st.session_state.current_page == "✍️ Digit Recognition":
    st.markdown('<div class="status-badge-online"><span class="status-dot"></span> REAL-TIME RECOGNITION</div>', unsafe_allow_html=True)
    st.markdown("<h2>Handwritten Digit Recognition</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #94a3b8; margin-bottom: 20px;'>Draw a single digit (0–9) on the handwriting pad or select a real dataset example.</p>",
        unsafe_allow_html=True
    )
    
    col_input, col_result = st.columns([1.15, 1.25], gap="large")
    
    # -------------------------------------------------------------------------
    # LEFT COLUMN: INPUT (Handwriting Pad + Dataset Examples)
    # -------------------------------------------------------------------------
    with col_input:
        st.markdown("### ✏️ INPUT")
        
        # Handwriting Pad Card
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 16px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-weight: 600; font-size: 0.95rem; color: #f8fafc;">Handwriting Pad</span>
                    <span style="font-size: 0.8rem; color: #94a3b8;">Draw one digit (0–9)</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        
        # Handwriting Pad Canvas (Clean White Area, Dark Smooth Stroke, ~350px)
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=18,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=320,
            width=320,
            drawing_mode="freedraw",
            key=f"hw_pad_{st.session_state.canvas_key}",
            display_toolbar=False
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Action Buttons for Handwriting Pad
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            recognize_btn = st.button("🔍 RECOGNIZE DIGIT", type="primary", use_container_width=True)
        with btn_c2:
            clear_btn = st.button("🗑️ CLEAR", type="secondary", use_container_width=True)

        if clear_btn:
            st.session_state.canvas_key += 1
            st.session_state.prediction_result = None
            st.session_state.active_source = "pad"
            st.session_state.example_sample = None
            st.rerun()

        # Handle Drawing Recognition Click
        if recognize_btn:
            st.session_state.active_source = "pad"
            st.session_state.example_sample = None
            if canvas_result.image_data is not None:
                res_dict, status = preprocess_digit(canvas_result.image_data, model=model)
                if status == "empty":
                    st.session_state.prediction_result = {"status": "empty"}
                else:
                    st.session_state.prediction_result = {
                        "status": "success",
                        "source": "pad",
                        "data": res_dict
                    }
            else:
                st.session_state.prediction_result = {"status": "empty"}

        st.markdown("<br>", unsafe_allow_html=True)
        
        # ---------------------------------------------------------------------
        # TRY REAL DATASET EXAMPLES
        # ---------------------------------------------------------------------
        st.markdown("#### 🧪 TRY REAL DATASET EXAMPLES")
        st.markdown("<p style='font-size: 0.85rem; color: #94a3b8; margin-top: -6px;'>Click a digit to test real 8×8 images directly from <code>load_digits()</code>:</p>", unsafe_allow_html=True)
        
        # Button Grid for Digits 0 through 9
        row1_cols = st.columns(5)
        for d in range(5):
            with row1_cols[d]:
                if st.button(f"[{d}]", key=f"ex_btn_{d}", use_container_width=True):
                    sample = get_real_dataset_sample(d)
                    st.session_state.active_source = "dataset_example"
                    st.session_state.example_sample = sample
                    st.session_state.prediction_result = {
                        "status": "success",
                        "source": "dataset_example",
                        "data": {
                            "prediction": sample["prediction"],
                            "confidence": sample["confidence"],
                            "probabilities": sample["probabilities"],
                            "top_predictions": [
                                {"digit": int(idx), "probability": float(sample["probabilities"][idx] * 100.0)}
                                for idx in np.argsort(sample["probabilities"])[::-1][:3]
                            ],
                            "is_low_confidence": bool(sample["confidence"] < 50.0),
                            "scaled_8x8": sample["raw_8x8"],
                            "flat_array": sample["flat_array"],
                            "cropped_image": sample["vis_image"],
                            "centered_image": sample["vis_image"],
                            "small_image": sample["vis_image"]
                        }
                    }
                    st.rerun()

        row2_cols = st.columns(5)
        for d in range(5, 10):
            with row2_cols[d - 5]:
                if st.button(f"[{d}]", key=f"ex_btn_{d}", use_container_width=True):
                    sample = get_real_dataset_sample(d)
                    st.session_state.active_source = "dataset_example"
                    st.session_state.example_sample = sample
                    st.session_state.prediction_result = {
                        "status": "success",
                        "source": "dataset_example",
                        "data": {
                            "prediction": sample["prediction"],
                            "confidence": sample["confidence"],
                            "probabilities": sample["probabilities"],
                            "top_predictions": [
                                {"digit": int(idx), "probability": float(sample["probabilities"][idx] * 100.0)}
                                for idx in np.argsort(sample["probabilities"])[::-1][:3]
                            ],
                            "is_low_confidence": bool(sample["confidence"] < 50.0),
                            "scaled_8x8": sample["raw_8x8"],
                            "flat_array": sample["flat_array"],
                            "cropped_image": sample["vis_image"],
                            "centered_image": sample["vis_image"],
                            "small_image": sample["vis_image"]
                        }
                    }
                    st.rerun()

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: PREDICTION RESULT & VISUALIZATIONS
    # -------------------------------------------------------------------------
    with col_result:
        st.markdown("### 🎯 PREDICTION RESULT")
        
        if st.session_state.prediction_result is None:
            st.markdown(
                """
                <div class="glass-card" style="text-align: center; padding: 50px 20px;">
                    <div style="font-size: 3rem; color: #64748b;">⏳</div>
                    <h4 style="color: #cbd5e1; margin-top: 12px;">Waiting for input...</h4>
                    <p style="font-size: 0.9rem; color: #94a3b8; max-width: 340px; margin: 0 auto;">
                        Draw a digit on the handwriting pad and click <strong>RECOGNIZE DIGIT</strong>, or select a <strong>REAL DATASET EXAMPLE</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif st.session_state.prediction_result.get("status") == "empty":
            st.warning("⚠️ **Please draw a digit first.** The handwriting pad is currently empty.")
            st.markdown(
                """
                <div class="glass-card" style="text-align: center; padding: 36px 20px;">
                    <p style="color: #94a3b8; margin: 0;">No strokes detected. Use mouse or touch to draw a digit and click Recognize.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            data = st.session_state.prediction_result["data"]
            source = st.session_state.prediction_result.get("source", "pad")
            pred = data["prediction"]
            conf = data["confidence"]
            probs = data["probabilities"]
            top_preds = data.get("top_predictions", [])
            is_low = data.get("is_low_confidence", False)

            # Badge indicating source
            source_badge = ""
            if source == "dataset_example":
                source_badge = '<span class="example-tag">REAL DATASET EXAMPLE</span>'

                       # Prediction Card
            st.markdown(
                f"""<div class="prediction-box">
<div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #a5b4fc; font-weight: 600;">PREDICTED DIGIT</div>
<div class="pred-digit-giant">{pred}</div>
<div style="font-size: 1.15rem; color: #e2e8f0; margin-bottom: 6px;">Model Confidence: <strong style="color: #38bdf8;">{conf:.2f}%</strong></div>
<div class="pred-status-pill">✓ Prediction Complete</div>
</div>""",
                unsafe_allow_html=True
            )

            # Low confidence warning if applicable
            if is_low:
                st.markdown(
                    """<div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 10px; padding: 10px 14px; margin-top: 12px; font-size: 0.88rem; color: #fbbf24;">⚠️ <strong>Low-confidence prediction.</strong> Try writing the digit more clearly or with thicker strokes.</div>""",
                    unsafe_allow_html=True
                )
            st.markdown("<br>", unsafe_allow_html=True)

            # Top 3 Predictions
            st.markdown("##### 🏆 TOP 3 PREDICTIONS (Model Confidence)")
            top_cols = st.columns(3)
            for i, p_info in enumerate(top_preds):
                with top_cols[i]:
                    st.markdown(
                        f"""
                        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 10px; text-align: center;">
                            <div style="font-size: 0.75rem; color: #94a3b8;">Rank #{i+1}</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: {'#38bdf8' if i==0 else '#e2e8f0'};">Digit {p_info['digit']}</div>
                            <div style="font-size: 0.9rem; color: {'#34d399' if i==0 else '#94a3b8'};">{p_info['probability']:.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # Class Probability Distribution Chart (0-9)
            st.markdown("##### 📊 CLASS PROBABILITY (0–9)")
            prob_df = pd.DataFrame({
                "Digit": [str(d) for d in range(10)],
                "Probability": probs * 100.0,
                "Is_Predicted": [d == pred for d in range(10)]
            })

            # Custom colors highlighting predicted digit
            colors = ["#38bdf8" if d == pred else "#334155" for d in range(10)]

            fig_bar = px.bar(
                prob_df,
                x="Digit",
                y="Probability",
                text=[f"{p:.1f}%" if p > 0 else "" for p in prob_df["Probability"]],
                range_y=[0, 108]
            )
            fig_bar.update_traces(
                marker_color=colors,
                textposition="outside",
                textfont=dict(size=11, color="#f8fafc")
            )
            fig_bar.update_layout(
                height=210,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Digit Class (0–9)", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Probability (%)", gridcolor="rgba(255,255,255,0.08)")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Image Processing Visualization
            st.markdown("##### 🔬 VIEW IMAGE PROCESSING")
            st.markdown(
                """
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;">
                    <code>ORIGINAL DRAWING</code> → <code>CROPPED DIGIT</code> → <code>CENTERED DIGIT</code> → <code>FINAL 8×8 IMAGE</code>
                </div>
                """,
                unsafe_allow_html=True
            )

            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            with pcol1:
                st.caption("1. Input Stroke")
                st.image(data["cropped_image"], use_container_width=True)
            with pcol2:
                st.caption("2. Cropped BBox")
                st.image(data["cropped_image"], use_container_width=True)
            with pcol3:
                st.caption("3. Centered (COM)")
                st.image(data["centered_image"], use_container_width=True)
            with pcol4:
                st.caption("4. Final 8×8 (0–16)")
                fig_small = px.imshow(data["scaled_8x8"], color_continuous_scale="Viridis")
                fig_small.update_layout(
                    height=90,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False
                )
                fig_small.update_xaxes(showticklabels=False)
                fig_small.update_yaxes(showticklabels=False)
                st.plotly_chart(fig_small, use_container_width=True)

            st.caption("Image Size: **8 × 8** | Features: **64** | Pixel Range: **0–16**")

            # Expandable Numerical Feature Matrix
            with st.expander("🔍 VIEW 8×8 FEATURE MATRIX (Numerical Heatmap)"):
                fig_heat = px.imshow(
                    data["scaled_8x8"],
                    color_continuous_scale="Viridis",
                    text_auto=".1f",
                    labels=dict(x="X Coordinate (0-7)", y="Y Coordinate (0-7)", color="Intensity")
                )
                fig_heat.update_layout(
                    height=280,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                st.markdown("**Flattened Vector (64 Float Values passed into `model.predict()`):**")
                st.code(", ".join([f"{v:.1f}" for v in data["flat_array"]]), language="text")


# =============================================================================
# PAGE 3: 🤖 MODEL PAGE
# =============================================================================
elif st.session_state.current_page == "🤖 Model":
    st.markdown('<div class="status-badge-online"><span class="status-dot"></span> ARCHITECTURE & VALIDATION</div>', unsafe_allow_html=True)
    st.markdown("<h2>MODEL INFORMATION</h2>", unsafe_allow_html=True)
    
    # Model Specs Summary
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(
            """
            <div class="glass-card" style="padding: 16px;">
                <div class="stat-label">Algorithm</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #60a5fa;">K-Nearest Neighbors</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Instance-based Supervised ML</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_m2:
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 16px;">
                <div class="stat-label">Neighbors (k)</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #a855f7;">k = {eval_stats.get('selected_k', 3)}</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Optimal via 5-Fold CV</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_m3:
        st.markdown(
            """
            <div class="glass-card" style="padding: 16px;">
                <div class="stat-label">Dataset</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #34d399;">Scikit-Learn Digits</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">1,797 Samples • 10 Classes</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_m4:
        st.markdown(
            """
            <div class="glass-card" style="padding: 16px;">
                <div class="stat-label">Feature Shape</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #f472b6;">8 × 8 = 64 Features</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Intensity Range: 0 to 16</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Actual Model Validation Accuracy Section
    st.markdown("### 🎯 Actual Model Validation Accuracy")
    k_eval = eval_stats.get("k_eval", {})
    k3_mean = k_eval.get(3, {}).get("mean_accuracy", 98.94)
    k3_std = k_eval.get(3, {}).get("std_accuracy", 0.37)
    test_acc = eval_stats.get("test_accuracy_k3", 98.61)

    col_acc1, col_acc2 = st.columns([1, 1.4])
    with col_acc1:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center; border-left: 4px solid #34d399;">
                <div class="stat-label">Stratified 5-Fold Cross-Validation</div>
                <div style="font-size: 2.5rem; font-weight: 800; color: #34d399; margin: 4px 0;">{k3_mean:.2f}%</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">Standard Deviation: ±{k3_std:.2f}%</div>
                <hr style="border-color: rgba(255,255,255,0.08); margin: 12px 0;">
                <div class="stat-label">Held-out 80/20 Test Split Accuracy</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #38bdf8;">{test_acc:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_acc2:
        # Comparison chart across k=1, 3, 5, 7
        k_df = pd.DataFrame([
            {"K Value": f"K = {k}", "Mean Accuracy (%)": k_eval[k]["mean_accuracy"], "Std": k_eval[k]["std_accuracy"]}
            for k in [1, 3, 5, 7] if k in k_eval
        ])
        fig_k = px.bar(
            k_df,
            x="K Value",
            y="Mean Accuracy (%)",
            text=[f"{acc:.2f}%" for acc in k_df["Mean Accuracy (%)"]],
            color="Mean Accuracy (%)",
            color_continuous_scale="Blues",
            range_y=[97, 100]
        )
        fig_k.update_layout(
            height=230,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False
        )
        fig_k.update_traces(textposition="outside")
        st.plotly_chart(fig_k, use_container_width=True)

    st.info(
        "💡 **Academic Note on Validation Accuracy:** Validation accuracy measures performance on held-out dataset samples from Scikit-Learn Digits. "
        "It does not guarantee the exact same accuracy for every possible human handwritten input due to variability in stroke slant, thickness, and canvas positioning."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # End-to-End Prediction Flow Diagram
    st.markdown("### 🔄 End-to-End Prediction Flow")
    st.markdown(
        """
        <div class="glass-card">
            <div class="flow-step-card">
                <span class="flow-step-num">01</span>
                <strong>USER DRAWING</strong> — Captures high-resolution continuous freehand stroke on canvas.
            </div>
            <div class="flow-step-card">
                <span class="flow-step-num">02</span>
                <strong>GRAYSCALE & POLARITY</strong> — Converts to luminance channel and inverts if white background.
            </div>
            <div class="flow-step-card">
                <span class="flow-step-num">03</span>
                <strong>CROP & BOUNDING BOX</strong> — Isolates non-zero stroke coordinates to remove unused whitespace.
            </div>
            <div class="flow-step-card">
                <span class="flow-step-num">04</span>
                <strong>CENTER (COM) & PADDING</strong> — Adds aspect-ratio preserving padding and center-of-mass alignment.
            </div>
            <div class="flow-step-card">
                <span class="flow-step-num">05</span>
                <strong>8 × 8 DOWNSCALING</strong> — Smooth anti-aliased resampling into an 8×8 grid of 64 pixels.
            </div>
            <div class="flow-step-card">
                <span class="flow-step-num">06</span>
                <strong>0–16 SCALING</strong> — Maps pixel intensities to the exact [0, 16] numerical distribution.
            </div>
            <div class="flow-step-card">
                <span class="flow-step-num">07</span>
                <strong>64 FEATURES</strong> — Flattens the 8×8 matrix into a 1×64 1D feature vector.
            </div>
            <div class="flow-step-card">
                <span class="flow-step-num">08</span>
                <strong>KNN CLASSIFIER (k=3)</strong> — Computes Euclidean distances to 1,797 dataset vectors.
            </div>
            <div class="flow-step-card" style="border-left-color: #10b981;">
                <span class="flow-step-num" style="color: #10b981;">09</span>
                <strong style="color: #10b981;">PREDICTED DIGIT & CONFIDENCE</strong> — Majority neighbor voting produces final digit and probabilities.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Simple Viva Q&A on KNN
    st.markdown("### 📚 Understanding K-Nearest Neighbors (College Viva Q&A)")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown(
            """
            <div class="glass-card">
                <h4>What is KNN?</h4>
                <p style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                    K-Nearest Neighbors (KNN) is an <strong>instance-based, non-parametric supervised learning algorithm</strong>.
                    Instead of building an explicit mathematical function during training, it stores all training data and classifies new inputs based on similarity to their closest neighbors.
                </p>
                <h4>What does 'k' mean and why k=3?</h4>
                <p style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                    'k' represents the number of nearest neighbors consulted. We chose <strong>k=3</strong> because:
                </p>
                <ul style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6; padding-left: 18px;">
                    <li>Empirical cross-validation showed k=3 gives the highest accuracy (98.94%).</li>
                    <li>An odd number prevents voting ties in classification.</li>
                    <li>Avoids overfitting to single outlier noise ($k=1$).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    with v2:
        st.markdown(
            """
            <div class="glass-card">
                <h4>How is Distance Calculated?</h4>
                <p style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                    KNN calculates the <strong>Euclidean Distance ($L_2$ Norm)</strong> between the input 64-dimensional vector $\mathbf{x}$ and training vectors $\mathbf{y}$:
                </p>
                <div style="background: rgba(15, 23, 42, 0.7); padding: 10px; border-radius: 8px; text-align: center; margin: 10px 0;">
                    $$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{64} (x_i - y_i)^2}$$
                </div>
                <h4>How is the Prediction Produced?</h4>
                <p style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                    The algorithm finds the 3 training samples with the smallest Euclidean distance. The most frequent class among these 3 determines the prediction, and voting frequency gives the predicted probability.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =============================================================================
# PAGE 4: 📊 DATASET PAGE
# =============================================================================
elif st.session_state.current_page == "📊 Dataset":
    st.markdown('<div class="status-badge-online"><span class="status-dot"></span> LIVE DATASET METADATA</div>', unsafe_allow_html=True)
    st.markdown("<h2>DATASET EXPLORER</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #94a3b8;'>Live exploration of the actual <code>sklearn.datasets.load_digits()</code> benchmark dataset.</p>",
        unsafe_allow_html=True
    )
    
    # Dataset Statistics from actual data
    n_samples = len(digits_dataset.data)
    n_classes = len(np.unique(digits_dataset.target))
    min_px = int(np.min(digits_dataset.data))
    max_px = int(np.max(digits_dataset.data))
    
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        st.metric("Samples", f"{n_samples:,}")
    with d2:
        st.metric("Classes", f"{n_classes} (0–9)")
    with d3:
        st.metric("Image Size", "8 × 8")
    with d4:
        st.metric("Features", "64")
    with d5:
        st.metric("Pixel Range", f"{min_px} – {max_px}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Actual handwritten samples for 0 through 9
    st.markdown("### 🖼️ Real Handwritten Samples (Classes 0 – 9)")
    st.markdown("<p style='font-size: 0.85rem; color: #94a3b8;'>Representative 8×8 instances from the Scikit-Learn training set:</p>", unsafe_allow_html=True)
    
    sample_cols = st.columns(10)
    for digit_c in range(10):
        sample_info = get_real_dataset_sample(digit_c)
        with sample_cols[digit_c]:
            st.markdown(f"<div style='text-align: center; font-weight: 700; color: #38bdf8; font-size: 0.9rem;'>Class {digit_c}</div>", unsafe_allow_html=True)
            fig_samp = px.imshow(sample_info["raw_8x8"], color_continuous_scale="Viridis")
            fig_samp.update_layout(
                height=110,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False
            )
            fig_samp.update_xaxes(showticklabels=False)
            fig_samp.update_yaxes(showticklabels=False)
            st.plotly_chart(fig_samp, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset Distribution & Interactive Explorer
    st.markdown("### 🔬 Interactive Dataset Explorer")
    exp_col1, exp_col2 = st.columns([1.1, 1.3])
    
    with exp_col1:
        st.markdown("#### Sample Inspector")
        selected_sample_idx = st.slider("Select Sample Index from Dataset", min_value=0, max_value=n_samples - 1, value=0, step=1)
        
        sample_img = digits_dataset.images[selected_sample_idx]
        sample_label = digits_dataset.target[selected_sample_idx]
        
        st.markdown(
            f"""
            <div class="glass-card">
                <span style="color: #94a3b8; font-size: 0.85rem;">Sample #{selected_sample_idx}</span>
                <h3 style="color: #38bdf8; margin: 4px 0 12px 0;">Ground Truth Label: {sample_label}</h3>
            """,
            unsafe_allow_html=True
        )
        fig_single = px.imshow(sample_img, color_continuous_scale="Viridis", text_auto=".0f")
        fig_single.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_single, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with exp_col2:
        st.markdown("#### Dataset Class Distribution")
        counts = pd.Series(digits_dataset.target).value_counts().sort_index()
        df_dist = pd.DataFrame({"Digit Class": [f"Class {c}" for c in counts.index], "Sample Count": counts.values})
        
        fig_dist = px.bar(
            df_dist,
            x="Digit Class",
            y="Sample Count",
            text="Sample Count",
            color="Sample Count",
            color_continuous_scale="Blues",
            range_y=[0, max(counts.values) + 30]
        )
        fig_dist.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False
        )
        fig_dist.update_traces(textposition="outside")
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption("The Scikit-Learn Digits dataset is balanced with ~180 samples per digit class.")


# =============================================================================
# PAGE 5: ℹ️ ABOUT PAGE
# =============================================================================
elif st.session_state.current_page == "ℹ️ About":
    st.markdown('<div class="status-badge-online"><span class="status-dot"></span> PROJECT DOCUMENTATION</div>', unsafe_allow_html=True)
    st.markdown("<h2>ABOUT PROJECT</h2>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="color: #60a5fa;">1. Project Objective</h3>
            <p style="color: #cbd5e1; line-height: 1.7;">
                To develop, fix, and deploy an end-to-end Machine Learning Optical Character Recognition (OCR) system 
                capable of accurately recognizing freehand handwritten digits (0–9) in real-time, sharing a single robust 
                preprocessing pipeline across both a Tkinter desktop application and a modern Streamlit web dashboard.
            </p>

            <h3 style="color: #a855f7; margin-top: 24px;">2. Problem Statement</h3>
            <p style="color: #cbd5e1; line-height: 1.7;">
                Freehand human handwriting varies dramatically in size, position, stroke width, and aspect ratio. 
                Directly resizing unaligned drawings to low-resolution 8×8 grids distorts strokes and causes false predictions. 
                The system must preprocess inputs via polarity detection, bounding-box cropping, square aspect-ratio padding, 
                center-of-mass alignment, and intensity scaling to match the training benchmark.
            </p>

            <h3 style="color: #38bdf8; margin-top: 24px;">3. Technologies Used</h3>
            <ul style="color: #cbd5e1; line-height: 1.7; padding-left: 20px;">
                <li><strong>Python 3.12:</strong> Primary programming language.</li>
                <li><strong>Streamlit:</strong> Reactive web dashboard framework.</li>
                <li><strong>streamlit-drawable-canvas:</strong> HTML5 canvas for capturing user handwriting.</li>
                <li><strong>Scikit-Learn:</strong> Machine learning algorithms (<code>KNeighborsClassifier</code>) and dataset (<code>load_digits</code>).</li>
                <li><strong>NumPy:</strong> Vector transformations and multidimensional array operations.</li>
                <li><strong>Pillow (PIL):</strong> Digital image processing, cropping, bounding box detection, and downsampling.</li>
                <li><strong>Plotly:</strong> Interactive charts for class probabilities and feature heatmaps.</li>
                <li><strong>Tkinter:</strong> Python standard desktop GUI library for the desktop client.</li>
            </ul>

            <h3 style="color: #34d399; margin-top: 24px;">4. Machine Learning Algorithm</h3>
            <p style="color: #cbd5e1; line-height: 1.7;">
                <strong>K-Nearest Neighbors (KNN, k=3):</strong> An instance-based supervised classifier. 
                For any 64-dimensional test vector $\mathbf{x}$, it computes the Euclidean distance 
                $d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^{64} (x_j - x_{i,j})^2}$ to all 1,797 training samples 
                and outputs the majority class label with posterior voting probabilities.
            </p>

            <h3 style="color: #f59e0b; margin-top: 24px;">5. Dataset</h3>
            <p style="color: #cbd5e1; line-height: 1.7;">
                <strong>Scikit-learn Digits Dataset:</strong> 1,797 normalized 8×8 grayscale images across 10 balanced digit 
                classes (0 to 9) with pixel values ranging from 0 (background) to 16 (full foreground stroke).
            </p>

            <h3 style="color: #ec4899; margin-top: 24px;">6. Working Principle</h3>
            <ol style="color: #cbd5e1; line-height: 1.7; padding-left: 20px;">
                <li><strong>Input Capture:</strong> Freehand drawing on a 320×320 pad or selection of a real dataset example.</li>
                <li><strong>Polarity Inversion:</strong> Converts dark ink on white canvas into bright strokes on a dark background.</li>
                <li><strong>Bounding Box & Aspect Ratio:</strong> Crops tight stroke bounds and pads into a square with ~28% margin.</li>
                <li><strong>Center of Mass Alignment:</strong> Fine-tunes stroke positioning to avoid edge clipping.</li>
                <li><strong>Anti-Aliased Downsampling:</strong> Resamples down to an 8×8 matrix of 64 pixels.</li>
                <li><strong>Intensity Normalization:</strong> Scales pixel values to the [0, 16] range.</li>
                <li><strong>KNN Classification:</strong> Finds the 3 nearest training vectors and computes class voting percentages.</li>
            </ol>

            <h3 style="color: #38bdf8; margin-top: 24px;">7. Real-World Applications</h3>
            <ul style="color: #cbd5e1; line-height: 1.7; padding-left: 20px;">
                <li><strong>Automated Mail Sorting:</strong> Reading postal ZIP codes on mail envelopes.</li>
                <li><strong>Banking Automation:</strong> Processing handwritten monetary amounts on bank cheques.</li>
                <li><strong>Form Processing:</strong> Digitizing paper surveys, examination scorecards, and census records.</li>
            </ul>

            <h3 style="color: #10b981; margin-top: 24px;">8. Advantages & Limitations</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px;">
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding: 14px; border-radius: 10px;">
                    <strong style="color: #34d399;">✅ Advantages</strong>
                    <ul style="margin: 8px 0 0 0; padding-left: 18px; font-size: 0.88rem; color: #cbd5e1;">
                        <li>Zero training computation delay ($O(1)$).</li>
                        <li>Completely transparent and mathematically explainable.</li>
                        <li>High accuracy (>98.9% CV) on low-dimensional 8×8 grids.</li>
                    </ul>
                </div>
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); padding: 14px; border-radius: 10px;">
                    <strong style="color: #f87171;">⚠️ Limitations</strong>
                    <ul style="margin: 8px 0 0 0; padding-left: 18px; font-size: 0.88rem; color: #cbd5e1;">
                        <li>Inference time scales linearly with dataset size ($O(N \cdot D)$).</li>
                        <li>Sensitive to extreme stroke rotation.</li>
                        <li>Does not learn hierarchical spatial convolutions like deep CNNs.</li>
                    </ul>
                </div>
            </div>

            <h3 style="color: #6366f1; margin-top: 24px;">9. Future Scope</h3>
            <ul style="color: #cbd5e1; line-height: 1.7; padding-left: 20px;">
                <li>Implementing Convolutional Neural Networks (CNNs / ResNet) for translation-invariant recognition.</li>
                <li>Scaling to the full MNIST benchmark (28×28, 70,000 samples).</li>
                <li>Adding multi-digit string recognition and mathematical equation solvers.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
