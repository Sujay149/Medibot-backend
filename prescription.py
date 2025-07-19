from PIL import Image
import google.generativeai as genai
from io import BytesIO

def analyze_prescription(image_bytes, api_key, image_type="Prescription Receipt"):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    image = Image.open(BytesIO(image_bytes))

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

    response = model.generate_content([prompt, image])
    return response.text
