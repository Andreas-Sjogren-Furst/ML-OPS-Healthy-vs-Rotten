"""
Streamlit frontend application to interact with the Healthy vs. Rotten FastAPI backend.
Upload images and receive classification results (healthy vs. rotten).
"""

import streamlit as st
import requests
from io import BytesIO


def main():
    """
    The main function for the Streamlit frontend application. It sets up
    the UI, handles file uploads, sends images to the backend API for
    classification, and displays the results.
    """
    st.set_page_config(page_title="Healthy vs. Rotten Classifier", layout="wide", initial_sidebar_state="expanded")

    st.markdown(
        """
        <h1 style='text-align: center; margin-bottom: 1rem;'>
            Healthy vs. Rotten Classifier
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.write("Upload one or more fruit images to classify them as **healthy** or **rotten**.")

    # pylint: disable=invalid-name
    API_ENDPOINT = "https://ml-healthy-vs-rotten-api-63364934645.europe-west1.run.app/predict"

    uploaded_files = st.file_uploader("Upload fruit images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        images_in_memory = []
        for file in uploaded_files:
            file_bytes = file.read()
            images_in_memory.append({"name": file.name, "type": file.type, "bytes": file_bytes})
            file.seek(0)

        if st.button("Predict"):
            files_data = [("files", (img["name"], BytesIO(img["bytes"]), img["type"])) for img in images_in_memory]

            with st.spinner("Sending images for prediction..."):
                try:
                    response = requests.post(API_ENDPOINT, files=files_data, timeout=10)
                    if response.status_code == 200 and response.headers.get("Content-Type") == "application/json":
                        predictions = response.json()
                        st.success("Predictions received!")

                        for img_data, prediction in zip(images_in_memory, predictions):
                            col1, col2 = st.columns([1, 2])

                            with col1:
                                st.image(BytesIO(img_data["bytes"]), caption=img_data["name"], use_container_width=True)

                            with col2:
                                st.markdown(f"**File Name:** {prediction['filename']}")
                                st.markdown(f"**Score:** {prediction['score']:.4f}")
                                st.markdown(f"**Label:** {prediction['label']}")

                            st.markdown("---")
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")

                except requests.RequestException as e:
                    st.error(f"An API error occurred: {e}")


if __name__ == "__main__":
    main()
