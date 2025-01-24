# frontend.py
"""
Streamlit frontend application to interact with the Healthy vs. Rotten FastAPI backend.
Upload images and receive classification results (healthy vs. rotten).
"""

import streamlit as st
import requests
from io import BytesIO


def main():
    # Set page config for a cleaner, modern look
    st.set_page_config(page_title="Healthy vs. Rotten Classifier", layout="wide", initial_sidebar_state="expanded")

    # Custom header styling
    st.markdown(
        """
        <h1 style='text-align: center; margin-bottom: 1rem;'>
            Healthy vs. Rotten Classifier
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.write("Upload one or more fruit images to classify them as **healthy** or **rotten**.")

    # Replace this URL with the actual endpoint where your FastAPI app is running
    API_ENDPOINT = "https://ml-healthy-vs-rotten-api-63364934645.europe-west1.run.app/predict"

    uploaded_files = st.file_uploader("Upload fruit images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Store each image in memory for both display and API request
    if uploaded_files:
        images_in_memory = []
        for file in uploaded_files:
            file_bytes = file.read()
            images_in_memory.append({"name": file.name, "type": file.type, "bytes": file_bytes})
            file.seek(0)  # reset the pointer if needed

        if st.button("Predict"):
            # Prepare files for POST request
            files_data = [("files", (img["name"], BytesIO(img["bytes"]), img["type"])) for img in images_in_memory]

            with st.spinner("Sending images for prediction..."):
                try:
                    response = requests.post(API_ENDPOINT, files=files_data)
                    if response.status_code == 200:
                        predictions = response.json()
                        st.success("Predictions received!")

                        # Display results using a modern layout
                        for img_data, prediction in zip(images_in_memory, predictions):
                            col1, col2 = st.columns([1, 2])

                            with col1:
                                # Display the image; use_container_width replaces use_column_width
                                st.image(BytesIO(img_data["bytes"]), caption=img_data["name"], use_container_width=True)

                            with col2:
                                # Display classification results
                                st.markdown(f"**File Name:** {prediction['filename']}")
                                st.markdown(f"**Score:** {prediction['score']:.4f}")
                                st.markdown(f"**Label:** {prediction['label']}")

                            st.markdown("---")
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")

                except Exception as e:
                    st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
