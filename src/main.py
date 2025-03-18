import streamlit as st
from image_detection import extract_number_plate_text
from PIL import Image

st.title("License Plate Recognition App 🚗")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)
    
    plate_text = extract_number_plate_text(image)
    st.subheader("Extracted Number Plate: ")
    st.write(plate_text)