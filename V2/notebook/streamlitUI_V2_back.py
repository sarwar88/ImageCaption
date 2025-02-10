import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import requests
import tempfile
import os
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Image Captioning Application", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
        .main-title {
            font-size: 42px;
            font-weight: bold;
            color: black;
            text-align: center;
            padding: 20px 0;
            border-bottom: 2px solid #ddd;
        }
        .nav-container {
            display: flex;
            justify-content: center;
            background-color: #fff;
            padding: 5px 0;
            margin-bottom: 10px;
        }
        .caption-text {
            font-size: 20px;
            font-weight: bold;
            color: blue;
            text-align: center;
            margin-bottom: 10px;
        }
        .image-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin-top: 10px;
        }
        .image-container img {
            max-width: 100px;
            width: 20%;  /* Resize image to 50% of the container width */
            height: auto;  /* Maintain aspect ratio */
            border-radius: 10px;
            margin: auto;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<p class="main-title">Image Captioning Application</p>', unsafe_allow_html=True)

# --- NAVIGATION BAR ---
st.markdown('<div class="nav-container">', unsafe_allow_html=True)

cols = st.columns(4, gap="small")
pages = [
    {"name": "Home", "icon": "🏠"},
    {"name": "Data Visualization", "icon": "📈"},
    {"name": "Model Prediction", "icon": "📉"},
    {"name": "Model Performance", "icon": "📊"},
]

selected_page = None

for i, page in enumerate(pages):
    if cols[i].button(f"{page['icon']} {page['name']}", key=page['name']):
        selected_page = page['name']
        st.session_state["selected_page"] = page['name']

if "selected_page" in st.session_state:
    selected_page = st.session_state["selected_page"]
else:
    selected_page = "Data Visualization"

st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE CONTENT ---
if selected_page == "Data Visualization":
    sub_tabs = ["Audio file Exploration", "Features Exploration"]
    selected_tab = st.radio("", sub_tabs, horizontal=True)

    if selected_tab == "Features Exploration":
        st.markdown('<button class="button-style">Show Audio features Graphs</button>', unsafe_allow_html=True)

        st.subheader("Harmonics and Perceptual")
        fig, ax = plt.subplots(figsize=(8, 3))
        x = np.linspace(0, 2 * np.pi, 400)
        y1 = np.sin(x)
        y2 = np.cos(x)
        ax.plot(x, y1, 'purple')
        ax.plot(x, y2, 'orange')
        st.pyplot(fig)

elif selected_page == "Model Prediction":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                temp.write(uploaded_file.getvalue())
                temp.seek(0)
                temp_filename = temp.name

            # Make a request to the FastAPI backend
            with open(temp_filename, "rb") as file:
                files = {"file": file.read()}
                response = requests.post("http://localhost:8000/generate-caption/", files=files)
                response_data = response.json()
                caption = response_data.get("caption", "Error generating caption")

            # --- Centered Caption and Image in One Container ---
            st.markdown('<div class="image-container">', unsafe_allow_html=True)

            # Display the caption above the image, centered
            st.markdown(f'<p class="caption-text">{caption}</p>', unsafe_allow_html=True)

            # Display the image, centered with adjusted width
            st.image(temp_filename, use_container_width =True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Clean up
            os.remove(temp_filename)

        except Exception as e:
            st.error(f"Error: {str(e)}")
