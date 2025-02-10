from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

app = FastAPI(title="Image Captioning API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved model and required files
MODEL_PATH = "../output/model.keras"
TOKENIZER_PATH = "../output/tokenizer.pkl"
FEATURES_PATH = "../output/features.pkl"

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# Initialize feature extractor
feature_extractor = DenseNet201()
fe = Model(inputs=feature_extractor.input, outputs=feature_extractor.layers[-2].output)


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def extract_features(image):
    image = img_to_array(image)
    image = image / 255.
    image = np.expand_dims(image, axis=0)
    feature = fe.predict(image, verbose=0)
    return feature


def predict_caption(model, feature, tokenizer, max_length=34):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        y_pred = model.predict([feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)
        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break

    return in_text


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and process the uploaded image
    contents = await file.read()
    with open("temp_image.jpg", "wb") as f:
        f.write(contents)

    # Load and preprocess image
    image = load_img("temp_image.jpg", target_size=(224, 224))

    # Extract features
    feature = extract_features(image)

    # Generate caption
    caption = predict_caption(model, feature, tokenizer)

    # Clean up
    os.remove("temp_image.jpg")

    # Remove start and end tokens and clean up caption
    caption = caption.replace("startseq", "").replace("endseq", "").strip()

    return {"caption": caption}


@app.get("/")
async def root():
    return {"message": "Image Captioning API is running"}


if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
