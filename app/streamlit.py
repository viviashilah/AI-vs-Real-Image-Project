import streamlit as st
import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🤖",
    layout="wide"
)

# ======================================================
# LOAD CSS
# ======================================================

with open("app/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_ai_model():

    model = load_model(
        "model/mobilenetv2_model.keras",
        compile=False
    )

    return model


model = load_ai_model()

# ======================================================
# CONSTANT
# ======================================================

IMG_SIZE = 224

# ======================================================
# HEADER
# ======================================================

st.markdown("""
<h1>
AI vs Real Image Detector
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style="
    font-size:18px;
    color:#6b7280;
    margin-top:-10px;
    margin-bottom:35px;
">
    Deep Learning system for detecting AI-generated
    and real-world images using MobileNetV2.
</p>
""", unsafe_allow_html=True)

# ======================================================
# METRIC CARDS
# ======================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">98%</div>
        <div class="metric-label">
            Model Accuracy
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">2</div>
        <div class="metric-label">
            AI & Real Classes
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">
            MobileNetV2
        </div>
        <div class="metric-label">
            Transfer Learning
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:

    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">
            TensorFlow
        </div>
        <div class="metric-label">
            Deep Learning Framework
        </div>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# SPACING
# ======================================================

st.markdown("<br>", unsafe_allow_html=True)

# ======================================================
# MAIN LAYOUT
# ======================================================

left_col, right_col = st.columns([1.2, 1])

# ======================================================
# LEFT COLUMN
# ======================================================

with left_col:

    st.markdown("""
    <div class="custom-card">
        <h3>📤 Upload Image</h3>
        <p> Upload JPG or PNG image to classify whether
            the image is AI-generated or real-world.
        </p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ==================================================
    # SHOW IMAGE
    # ==================================================

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.markdown("<br>", unsafe_allow_html=True)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

# ======================================================
# RIGHT COLUMN
# ======================================================

with right_col:

    st.markdown("""
    <div class="result-box">
        <h3>🧠 Prediction Result</h3>
        <p> AI prediction and confidence score
            will appear here.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==================================================
    # PREDICTION
    # ==================================================

    if uploaded_file is not None:

        if st.button("Analyze Image"):

            with st.spinner("Analyzing image..."):

                # ======================================
                # PREPROCESS IMAGE
                # ======================================

                img = image.resize(
                    (IMG_SIZE, IMG_SIZE)
                )

                img = np.array(img)

                img = preprocess_input(img)

                img = np.expand_dims(
                    img,
                    axis=0
                )

                # ======================================
                # PREDICT
                # ======================================

                prediction = model.predict(
                    img,
                    verbose=0
                )[0][0]

                confidence = (
                    prediction
                    if prediction > 0.5
                    else 1 - prediction
                )

            # ==========================================
            # RESULT BADGE
            # ==========================================

            if prediction > 0.5:

                st.markdown("""
                <div class="real-badge">
                    ✅ REAL IMAGE
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="ai-badge">
                    ⚠ AI GENERATED IMAGE
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ==========================================
            # METRIC
            # ==========================================

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2%}"
            )

            # ==========================================
            # PROGRESS BAR
            # ==========================================

            st.progress(
                float(confidence)
            )

            # ==========================================
            # CONFIDENCE TEXT
            # ==========================================

            st.markdown(
                f"""
                <p style="
                    text-align:center;
                    color:#6b7280;
                    margin-top:10px;
                    font-size:15px;
                ">
                    Model confidence:
                    <b>{confidence:.2%}</b>
                </p>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "Upload an image to start prediction."
        )

# ======================================================
# FOOTER
# ======================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.divider()

st.markdown("""
<p style="
    text-align:center;
    color:#9ca3af;
    font-size:14px;
">
    Built with TensorFlow, Streamlit,
    and MobileNetV2
</p>
""", unsafe_allow_html=True)