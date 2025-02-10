import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="Image Captioning App",
    page_icon="🖼️",
    layout="wide"
)

def main():
    st.title("Image Captioning Application")
    st.write("Upload an image and get its description!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Make prediction
        with col2:
            st.subheader("Generated Caption")
            with st.spinner("Generating caption..."):
                # Prepare the file for the API request
                files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
                
                try:
                    # Make request to FastAPI endpoint
                    response = requests.post("http://localhost:8000/predict", files=files)
                    
                    if response.status_code == 200:
                        caption = response.json()["caption"]
                        st.success(caption)
                        
                        # Add some metrics or additional information
                        st.metric("Caption Length", len(caption.split()))
                    else:
                        st.error("Error generating caption. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the API. Please make sure the FastAPI server is running.")

if __name__ == "__main__":
    main()
