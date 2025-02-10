# fastapi_server.py (FastAPI for backend)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
import uvicorn
import io

# Load models and tokenizer at startup
MODEL_PATH = "../model/model.keras"
TOKENIZER_PATH = "../model/tokenizer.pkl"
FEATURE_EXTRACTOR_PATH = "../model/feature_extractor.keras"
IMG_SIZE = 224
MAX_LENGTH = 34

# Load resources
try:
    caption_model = load_model(MODEL_PATH)
    feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    print("Error loading models or tokenizer:", e)

app = FastAPI()

# Allow cross-origin requests for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate-caption/")
async def generate_caption(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        contents = await file.read()
        img = img_to_array(load_img(io.BytesIO(contents), target_size=(IMG_SIZE, IMG_SIZE))) / 255.0
        img = np.expand_dims(img, axis=0)
        image_features = feature_extractor.predict(img, verbose=0)

        # Generate caption
        in_text = "startseq"
        for _ in range(MAX_LENGTH):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
            yhat = caption_model.predict([image_features, sequence], verbose=0)
            yhat_index = np.argmax(yhat)
            word = tokenizer.index_word.get(yhat_index, None)
            if word is None:
                break
            in_text += " " + word
            if word == "endseq":
                break
        caption = in_text.replace("startseq", "").replace("endseq", "").strip()

        return {"caption": caption}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)