import streamlit as st
import numpy as np
import cv2
from script import main

st.set_page_config(page_title="Realistic Flag Mapper", layout="centered")

st.title("🎌 Realistic Waving Flag Generator")
st.write("Upload a white flag and a pattern to generate a realistic waving flag.")

# File uploads
pattern_file = st.file_uploader("Upload Pattern Image", type=["png", "jpg", "jpeg"])
flag_file = st.file_uploader("Upload White Flag Image", type=["png", "jpg", "jpeg"])

if st.button("Generate Waving Flag"):
    if not pattern_file or not flag_file:
        st.error("Please upload both images.")
    else:
        # Convert uploaded images to OpenCV format
        pattern_bytes = np.frombuffer(pattern_file.read(), np.uint8)
        flag_bytes    = np.frombuffer(flag_file.read(), np.uint8)

        pattern = cv2.imdecode(pattern_bytes, cv2.IMREAD_COLOR)
        flag    = cv2.imdecode(flag_bytes, cv2.IMREAD_COLOR)

        if pattern is None or flag is None:
            st.error("One of the images could not be read.")
        else:
            st.info("Processing… This may take a few seconds.")

            # Run pipeline (returns BGR)
            result = main(flag_img=flag, pattern_img=pattern)

            st.success("Done!")
            st.image(result, channels="BGR", caption="Final Realistic Flag")
            st.download_button(
                label="Download Result",
                data=cv2.imencode(".jpg", result)[1].tobytes(),
                file_name="Output.jpg",
                mime="image/jpeg"
            )
