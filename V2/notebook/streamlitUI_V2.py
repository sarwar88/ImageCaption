import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import requests
import tempfile
import os
import pickle
import pandas as pd
import json
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from textwrap import wrap

# --- MODEL CONFIGURATION ---
MODEL_PATH = "../model/model.keras"
TOKENIZER_PATH = "../model/tokenizer.pkl"
FEATURE_EXTRACTOR_PATH = "../model/feature_extractor.keras"
HISTORY_PATH = "../model/history.pkl"
JSON_PATH = "../model/bleu_scores.json"
IMG_SIZE = 224
MAX_LENGTH = 34

# Load tokenizer
try:
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(HISTORY_PATH, "rb") as f:
        loaded_history = pickle.load(f)
    # Load BLEU scores from JSON file
    with open(JSON_PATH, "rb") as f:
        bleu_scores = json.load(f)
except Exception as e:
    print("Error loading models or pickle files:", e)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Image Captioning Application", layout="centered")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
        .main-title {
            font-size: 42px;
            font-weight: bold;
            color: black;
            text-align: center;
            padding: 10px 0;
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
            max-width: 80px;
            width: 15%;
            height: auto;
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
    {"name": "Model Evaluation", "icon": "📊"},
]

# Function to calculate and display Blue Score
# Convert BLEU scores to DataFrame
blue_score_df = pd.DataFrame({"N Gram": ["Unigram", "Bigram", "Trigram", "Four-gram"],
                              "Blue Score": [bleu_scores["BLEU-1"], bleu_scores["BLEU-2"], bleu_scores["BLEU-3"], bleu_scores["BLEU-4"]]})


# Apply color styling to the DataFrame with conditional coloring based on BLEU scores
def color_row(row):
    # If the value in 'Blue Score' is numeric, we apply row styling
    if row["Blue Score"] > 0.6:
        return ['background-color: green'] * len(row)  # High BLEU score, green for entire row
    elif row["Blue Score"] > 0.4:
        return ['background-color: lightgreen'] * len(row)  # Medium BLEU score, yellow for entire row
    else:
        return ['background-color: lightyellow'] * len(row)  # Low BLEU score, red for entire row

# Apply the styling to the whole DataFrame row-wise
styled_blue_score_df = blue_score_df.style.apply(color_row, axis=1)

selected_page = None

for i, page in enumerate(pages):
    if cols[i].button(f"{page['icon']} {page['name']}", key=page['name']):
        selected_page = page['name']
        st.session_state["selected_page"] = page['name']

if "selected_page" in st.session_state:
    selected_page = st.session_state["selected_page"]
else:
    selected_page = "Home"

st.markdown('</div>', unsafe_allow_html=True)

# Utility function to read and process image
def read_image(path, img_size=224):
    img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
    img = img_to_array(img)
    img = img / 255.0
    return img

# Function to display images along with captions
def display_images(temp_df):
    temp_df = temp_df.reset_index(drop=True)
    plt.figure(figsize=(15, 15))  # Reduced figure size
    n = 0
    for i in range(10):
        n += 1
        plt.subplot(5, 5, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjusted spacing
        image = read_image(f"../input/flickr8k/Images/{temp_df.image[i]}")
        plt.imshow(image)
        plt.title("\n".join(wrap(temp_df.caption[i], 20)))
        plt.axis("off")
    st.pyplot(plt)

# Load dataset
data = pd.read_csv("../input/flickr8k/captions.txt")

# --- PAGE CONTENT ---
if selected_page == "Data Visualization":
    sub_tabs = st.tabs(["Text Features Exploration", "Image Features Exploration"])

    with sub_tabs[0]:
        if st.button("Show Top 20 Most Frequent Words"):
            word_counts = pd.DataFrame.from_dict(tokenizer.word_counts, orient='index', columns=['count']).sort_values(
                by='count', ascending=False).head(20)
            word_counts = pd.DataFrame.from_dict(tokenizer.word_counts, orient='index', columns=['count']) \
                              .sort_values(by='count', ascending=False).iloc[2:].head(20)

            fig, ax = plt.subplots(figsize=(10, 5))  # Reduced figure size
            word_counts.plot(kind='bar', ax=ax)
            ax.set_title("Top 20 Most Frequent Words in Captions")
            st.pyplot(fig)

    with sub_tabs[1]:
        st.subheader("Image Features Exploration")
        if st.button("Display Images Along With Captions"):
            display_images(data.sample(10))

elif selected_page == "Model Prediction":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                temp.write(uploaded_file.getvalue())
                temp.seek(0)
                temp_filename = temp.name

            with open(temp_filename, "rb") as file:
                files = {"file": file.read()}
                response = requests.post("http://localhost:8000/generate-caption/", files=files)
                response_data = response.json()
                caption = response_data.get("caption", "Error generating caption")

            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.markdown(f'<p class="caption-text">{caption}</p>', unsafe_allow_html=True)
            st.image(temp_filename, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            os.remove(temp_filename)
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif selected_page == "Model Evaluation":
    sub_tabs = st.tabs(["Learning Curve", "Blue Score"])

    with sub_tabs[0]:
        if st.button("Learning Curve Loss vs Epoch"):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(loaded_history['loss'], label='train')
            ax.plot(loaded_history['val_loss'], label='val')
            ax.set_title('Model Loss')
            ax.set_ylabel('Loss')
            ax.set_xlabel('Epoch')
            ax.legend()
            st.pyplot(fig)

    with sub_tabs[1]:
        if st.button("Display N Gram Blue Score"):
            st.dataframe(styled_blue_score_df, use_container_width=True)
