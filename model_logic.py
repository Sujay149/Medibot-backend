import os
import openai
import whisper
import faiss
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from gtts import gTTS
from tempfile import NamedTemporaryFile
from googletrans import Translator

from openai import OpenAI
client = OpenAI()

# === Load Data ===
dis_desc_df = pd.read_csv("dis_descp.csv")
updated_data_df = pd.read_csv("updated_disease_data.csv")
serious_disease_df = pd.read_csv("serious_diseases.csv")

translator = Translator()
whisper_model = whisper.load_model("base")

# === Setup FAISS ===
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def load_embeddings():
    from openai import OpenAI
    client = OpenAI()
    return client.embeddings.create

def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(input=chunk, model="text-embedding-3-small")
        vector = response.data[0].embedding
        embeddings.append(vector)
    return embeddings

# Load PDFs & create FAISS index
pdf1_text = extract_text_from_pdf("TheCureForAllDiseases.pdf")
pdf2_text = extract_text_from_pdf("Harrison.pdf")
all_chunks = split_into_chunks(pdf1_text) + split_into_chunks(pdf2_text)

chunk_texts = all_chunks
chunk_embeddings = embed_chunks(chunk_texts)

dimension = len(chunk_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings).astype("float32"))

def get_relevant_chunks(query, top_k=3):
    query_emb = client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
    D, I = index.search(np.array([query_emb]).astype("float32"), top_k)
    return [chunk_texts[i] for i in I[0]]

# === Transcription ===
def transcribe_audio_file(file_obj, lang='en'):
    with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        file_obj.save(temp_audio.name)
        result = whisper_model.transcribe(temp_audio.name)
        return result["text"]

# === Language Translation ===
def translate(text, lang='en', direction='to_en'):
    if lang == 'en':
        return text
    if direction == 'to_en':
        return translator.translate(text, dest='en').text
    else:
        return translator.translate(text, dest=lang).text

# === Disease Prediction (basic symptom matching) ===
def predict_disease(user_input):
    symptoms = user_input.lower().split(", ")
    matching_rows = updated_data_df[updated_data_df['Symptoms'].str.lower().apply(lambda x: all(symptom in x for symptom in symptoms))]
    if not matching_rows.empty:
        return matching_rows.iloc[0]["Disease"]
    return None

# === LLM Response ===
def generate_response(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are MediBot, a helpful AI health assistant. Use the context below to answer the user query.

Context:
{context}

Query: {query}
Answer:"""
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful health assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content.strip()

# === TTS ===
def generate_audio(text, lang='en'):
    speech = gTTS(text=text, lang=lang)
    filename = f"audio_{np.random.randint(10000)}.mp3"
    filepath = os.path.join("audio", filename)
    os.makedirs("audio", exist_ok=True)
    speech.save(filepath)
    return filepath

# === Main processing ===
def process_query(user_input, lang='en', model='gpt-3.5-turbo'):
    query_en = translate(user_input, lang=lang, direction='to_en')
    disease = predict_disease(query_en)

    if disease:
        context_chunks = [f"Disease prediction based on symptoms: {disease}"]
    else:
        context_chunks = get_relevant_chunks(query_en)

    llm_response_en = generate_response(query_en, context_chunks)
    llm_response_translated = translate(llm_response_en, lang=lang, direction='to_local')

    audio_path = generate_audio(llm_response_translated, lang=lang)

    return llm_response_translated, audio_path
