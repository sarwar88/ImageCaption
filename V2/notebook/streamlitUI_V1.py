import streamlit as st
import requests
import tempfile
import os

st.title("Image Captioning App")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        # Create a temporary file to store the uploaded image without exposing the file name
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

            # Display the caption without showing the filename
            st.markdown(f"<h3 style='text-align: center; color: blue;'>{caption}</h3>", unsafe_allow_html=True)

            # Display the image without filename or uploaded number
            st.image(uploaded_file, use_container_width =True)

        # Clean up the temporary file
        os.remove(temp_filename)
    except Exception as e:
        st.error(f"Error: {str(e)}")
