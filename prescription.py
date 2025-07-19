import streamlit as st
from PIL import Image
import google.generativeai as genai

# Streamlit page setup
st.set_page_config(page_title="🧾 Smart Medicine Analyzer", layout="centered")
st.title("💊 Medicine Analysis using Gemini 1.5 Flash")

# Sidebar: API Key
st.sidebar.header("🔐 Gemini API Key")
api_key = st.sidebar.text_input("Paste your Gemini API Key:", type="password")

# Configure Gemini
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Image upload
    uploaded_image = st.file_uploader("📤 Upload Prescription or Tablet Sheet Image", type=["jpg", "jpeg", "png"])

    # Type selector
    image_type = st.radio(
        "🔍 What kind of image is this?",
        ["Prescription Receipt", "Tablet Strip"],
        index=0
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Define prompt based on image type
        if image_type == "Prescription Receipt":
            prompt = (
                "The image shows a medical prescription or receipt. Extract the list of medicines, their usage or purpose, "
                "and the patient name if visible. Format the response in a clean sequence:\n\n"
                "1. Medicine Name\n   - Purpose: [Explain what it's used for]\n   - Dosage/Frequency (if available)\n\n"
                "Also include the doctor’s name or hospital if visible."
            )
        else:
            prompt = (
                "This is an image of a tablet strip or medicine packaging. Extract any visible text or information. "
                "If there’s useful information like medicine name, usage, brand, or expiry, summarize it meaningfully. "
                "If the image has only partial text, make a sensible attempt to interpret it or explain what is visible."
            )

        with st.spinner("🔍 Analyzing image with Gemini..."):
            try:
                response = model.generate_content([prompt, image])
                extracted_info = response.text
                st.success("✅ Analysis Complete")
                st.subheader("📋 Extracted Information")
                st.markdown(extracted_info)
            except Exception as e:
                st.error(f"Error while processing: {e}")
else:
    st.warning("Please enter your Gemini API key in the sidebar.")
