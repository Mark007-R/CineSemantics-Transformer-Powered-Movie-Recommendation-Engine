import streamlit as st
from PIL import Image
import tempfile
import os
import sys
import pandas as pd
from pathlib import Path

utils_dir = Path(__file__).resolve().parent.parent / 'utils'
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))

import config
from image_main import search_similar_movies

st.title("Image-Based Movie Recommendation")
st.write("Upload a movie poster image to get similar movie recommendations.")

uploaded_file = st.file_uploader("Choose a movie poster image", type=["jpg", "jpeg", "png", "webp", "bmp", "gif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Poster", width=300)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        temp_image_path = tmp_file.name

    st.write("Searching for similar movies...")
    try:
        results = search_similar_movies(temp_image_path)
        if results and len(results) > 0:
            st.subheader("Recommended Movies:")
            for movie in results:
                st.markdown(f"**{movie.get('title', 'Unknown Title')}**")
                img_path = movie.get('image_path')
                if img_path and os.path.exists(img_path):
                    st.image(img_path, width=200)
                else:
                    st.write("[Image not found]")
                st.write(f"Similarity: {movie.get('similarity_percent', '')}")
                st.write("---")
        else:
            st.warning("No recommendations found.")
    except Exception as e:
        st.error(f"Error during recommendation: {e}")
    finally:
        os.remove(temp_image_path)
else:
    st.info("Please upload a movie poster image to get recommendations.")
