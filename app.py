import os
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from config import SUPPORTED_FORMATS
from styles import load_css
from utils import (
    detect_objects,
    read_text,
    speak,
    load_uploaded_image,
    detection_summary,
    save_image,
    get_history,
)
from video_processor import VideoProcessor

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------

st.set_page_config(
    page_title="AI Accessibility Assistant",
    page_icon="♿",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

# -----------------------------------------------------
# Session State
# -----------------------------------------------------

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.45

if "voice" not in st.session_state:
    st.session_state.voice = True

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

logo = "images/logo.png"

if os.path.exists(logo):
    st.sidebar.image(logo, width=120)

st.sidebar.title("♿ AI Accessibility Assistant")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📹 Live Camera",
        "🖼 Upload Image",
        "📖 OCR Reader",
        "📜 Detection History",
        "⚙ Settings",
        "ℹ About",
    ],
)

st.sidebar.markdown("---")

st.sidebar.success("YOLO11 + EasyOCR + Streamlit")

st.sidebar.markdown(
"""
### Features

- Live Detection
- OCR Reader
- Voice Assistant
- Detection History
- Image Upload
"""
)

# -----------------------------------------------------
# HOME
# -----------------------------------------------------

if page == "🏠 Home":

    st.markdown(
        "<h1 class='main-title'>AI Accessibility Assistant</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p class='sub-title'>Helping visually impaired users using Artificial Intelligence</p>",
        unsafe_allow_html=True,
    )

    banner = "images/banner.png"

    if os.path.exists(banner):
        st.image(banner, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
<div class='card'>
<h3>📹 Live Detection</h3>
Detect surrounding objects in real time.
</div>
""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
<div class='card'>
<h3>📖 OCR Reader</h3>
Extract printed text from images.
</div>
""",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
<div class='card'>
<h3>🔊 Voice Guidance</h3>
Offline speech feedback.
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader("Project Features")

    st.write("✅ Real-Time Object Detection")
    st.write("✅ OCR Text Recognition")
    st.write("✅ Voice Feedback")
    st.write("✅ Direction Detection")
    st.write("✅ Distance Estimation")
    st.write("✅ Detection History")

    st.success("System Ready")

# -----------------------------------------------------
# LIVE CAMERA
# -----------------------------------------------------

elif page == "📹 Live Camera":

    st.title("📹 Live Camera")

    st.info("Allow camera permission when prompted.")

    ctx = webrtc_streamer(
        key="camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )
# -----------------------------------------------------
# UPLOAD IMAGE
# -----------------------------------------------------

elif page == "🖼 Upload Image":

    st.title("🖼 Image Object Detection")

    uploaded = st.file_uploader(
        "Upload an Image",
        type=SUPPORTED_FORMATS,
        key="image_upload",
    )

    if uploaded is not None:

        try:

            image = load_uploaded_image(uploaded)

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True,
            )

            if st.button("🚀 Detect Objects"):

                with st.spinner("Running YOLO Detection..."):

                    result_image, detections = detect_objects(
                        image,
                        confidence=st.session_state.confidence,
                    )

                st.success("Detection Completed Successfully")

                st.image(
                    result_image,
                    caption="Detection Result",
                    use_container_width=True,
                )

                if detections:

                    df = pd.DataFrame(detections)

                    # Hide internal bounding box
                    if "box" in df.columns:
                        df = df.drop(columns=["box"])

                    st.subheader("Detected Objects")

                    st.dataframe(
                        df,
                        use_container_width=True,
                    )

                    st.subheader("Detection Summary")

                    st.json(
                        detection_summary(detections)
                    )

                    save_path = save_image(result_image)

                    with open(save_path, "rb") as file:

                        st.download_button(
                            label="📥 Download Result",
                            data=file,
                            file_name="detected_image.jpg",
                            mime="image/jpeg",
                        )

                    labels = [
                        obj["label"]
                        for obj in detections
                    ]

                    speak(
                        "Detected " + ", ".join(labels)
                    )

                else:

                    st.warning(
                        "No objects detected."
                    )

        except Exception as e:

            st.error(
                f"Detection Error : {e}"
            )


# -- -----------------------------------------------------
# -----------------------------------------------------
# OCR READER
# -----------------------------------------------------

elif page == "📖 OCR Reader":

    st.title("📖 OCR Text Reader")

    uploaded = st.file_uploader(
        "Upload an image",
        type=SUPPORTED_FORMATS,
        key="ocr_image"
    )

    # Initialize OCR text
    if "ocr_text" not in st.session_state:
        st.session_state["ocr_text"] = ""

    if uploaded is not None:

        try:

            image = load_uploaded_image(uploaded)

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True,
            )

            # Extract text button
            if st.button("🔍 Read Text"):

                with st.spinner("Extracting Text..."):

                    text = read_text(image)

                st.session_state["ocr_text"] = text

            # Show extracted text
            if st.session_state["ocr_text"]:

                st.success("Text Extracted Successfully")

                st.text_area(
                    "Detected Text",
                    value=st.session_state["ocr_text"],
                    height=250,
                )

                col1, col2 = st.columns(2)

                # -----------------------------
                # Speak Text
                # -----------------------------
                with col1:

                    if st.button("🔊 Speak Text"):

                        with st.spinner("Speaking..."):

                            speak(st.session_state["ocr_text"])

                        st.success("Voice played successfully.")

                # -----------------------------
                # Download Text
                # -----------------------------
                with col2:

                    st.download_button(
                        label="📥 Download Text",
                        data=st.session_state["ocr_text"],
                        file_name="ocr_result.txt",
                        mime="text/plain",
                    )

            else:

                st.info("Click '🔍 Read Text' to extract text.")

        except Exception as e:

            st.error(f"OCR Error: {e}")

    else:

        # Clear previous OCR text when no image is uploaded
        st.session_state["ocr_text"] = ""


# DETECTION HISTORY
# -----------------------------------------------------

elif page == "📜 Detection History":
    st.title("📜 Detection History")

    history = get_history()

    if not history:

        st.info("No detection history available.")

    else:

        rows = []

        for item in history:

            for det in item["detections"]:

                rows.append(
                    {
                        "Time": item["time"],
                        "Object": det["label"],
                        "Confidence (%)": round(det["confidence"] * 100, 2),
                        "Direction": det["direction"],
                        "Distance": det["distance"],
                    }
                )

        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            use_container_width=True,
        )

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download History",
            csv,
            file_name="detection_history.csv",
            mime="text/csv",
        )

# -----------------------------------------------------
# SETTINGS
# -----------------------------------------------------

elif page == "⚙ Settings":

    st.title("⚙ Settings")

    st.subheader("Detection")

    st.session_state.confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=st.session_state.confidence,
        step=0.05,
    )

    st.subheader("Voice")

    st.session_state.voice = st.checkbox(
        "Enable Voice Assistant",
        value=st.session_state.voice,
    )

    st.success("Settings updated successfully.")

# -----------------------------------------------------
# ABOUT
# -----------------------------------------------------

elif page == "ℹ About":

    st.title("ℹ AI Accessibility Assistant")

    st.markdown("""
### 🎯 Project Objective

AI Accessibility Assistant helps visually impaired users
understand their surroundings using Artificial Intelligence.

---

### 🚀 Features

- Real-Time Object Detection
- OCR Text Reader
- Voice Guidance
- Left / Center / Right Navigation
- Distance Estimation
- Detection History
- Image Upload Detection
- Beautiful Streamlit Dashboard

---

### 🛠 Technologies

- Python 3.12
- Streamlit
- Streamlit WebRTC
- YOLO11 (Ultralytics)
- EasyOCR
- OpenCV
- pyttsx3

---

### 👨‍💻 Developed For

AI-powered accessibility solution for assisting visually
impaired users in everyday environments.
""")

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------

st.markdown("---")

st.markdown(
    """
<div class="footer">

<b>AI Accessibility Assistant</b><br>

Built with ❤️ using Python, Streamlit, YOLO11 and EasyOCR.

<br><br>

© 2026 All Rights Reserved

</div>
""",
    unsafe_allow_html=True,
)