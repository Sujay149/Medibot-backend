# =================== BASE LOGIC ===================
import os
import uuid
import asyncio
import numpy as np
import pandas as pd
import fitz
import faiss
import whisper
import gradio as gr
from gtts import gTTS
from openai import OpenAI
from googletrans import Translator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Shared data loading
def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


# ========== FASTAPI API ENDPOINTS ========== #
from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# BASE API: Transcribe audio
@app.post("/base/transcribe/")
async def base_transcribe(file: UploadFile = File(...), lang: str = Form("en")):
    temp_path = f"temp_{uuid.uuid4().hex}.wav"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    text = await transcribe_audio_base(temp_path, lang)
    os.remove(temp_path)
    return {"transcription": text}

# BASE API: Medical Q&A
@app.post("/base/ask/")
async def base_ask(
    text: str = Form(...),
    lang: str = Form("en"),
    model: str = Form("llama3.2")
):
    response_text, audio_path = await handle_text_input_base(text, lang, model)
    return {"response": response_text, "audio_file": audio_path}

# PREMIUM API: Transcribe audio
@app.post("/premium/transcribe/")
async def premium_transcribe(file: UploadFile = File(...), lang: str = Form("en")):
    temp_path = f"temp_{uuid.uuid4().hex}.wav"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    text = await transcribe_audio_premium(temp_path, lang)
    os.remove(temp_path)
    return {"transcription": text}

# PREMIUM API: Medical Q&A
@app.post("/premium/ask/")
async def premium_ask(
    text: str = Form(...),
    lang: str = Form("en"),
    model: str = Form("ollama/llama3.2")
):
    response_text, audio_path = await handle_text_input_premium(text, lang, model)
    return {"response": response_text, "audio_file": audio_path}

# Serve generated audio files
@app.get("/audio/{filename}")
def get_audio(filename: str):
    return FileResponse(filename, media_type="audio/mpeg")

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to Medibot API. Use /base/ask, /premium/ask, /base/transcribe, /premium/transcribe."}
import os
import uuid
import asyncio
import numpy as np
import pandas as pd
import fitz
import faiss
import whisper
import gradio as gr
from gtts import gTTS
from openai import OpenAI
from googletrans import Translator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# ----------------- Initialize APIs -----------------
openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
whisper_model = whisper.load_model("tiny")
translator = Translator()

# ----------------- RAG: Data Loading -----------------
rag_df = pd.read_csv('updated_disease_data_13_fuzzy_filled (1).csv')
rag_df2 = pd.read_csv('serious_diseases.csv')
rag_df3 = pd.read_csv('dis_descp.csv')

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

cure_text = extract_text_from_pdf('TheCureForAllDiseases.pdf')
textbook_text = extract_text_from_pdf('Harrison.pdf')

rag_documents = []
for df in [rag_df, rag_df2, rag_df3]:
    for _, row in df.iterrows():
        doc_text = " | ".join(str(cell) for cell in row if pd.notna(cell))
        rag_documents.append(doc_text)
rag_documents += [p.strip() for p in cure_text.split('\n\n') if p.strip()]
rag_documents += [p.strip() for p in textbook_text.split('\n\n') if p.strip()]

# ----------------- FAISS Embeddings -----------------
def get_embedding(text):
    response = openai.embeddings.create(input=text, model="nomic-embed-text")
    return np.array(response.data[0].embedding, dtype=np.float32)

document_embeddings = []
batch_size = 16
for i in range(0, len(rag_documents), batch_size):
    for txt in rag_documents[i:i + batch_size]:
        document_embeddings.append(get_embedding(txt))

document_embeddings = np.vstack(document_embeddings)
embedding_dim = document_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(document_embeddings)

# ----------------- Core Logic -----------------
system_message = """You are a knowledgeable and compassionate medical AI assistant..."""

async def translate_text(text, target_lang='en'):
    result = await asyncio.to_thread(translator.translate, text, dest=target_lang)
    return result.text

async def transcribe_audio(file_path, user_lang):
    try:
        result = whisper_model.transcribe(file_path)
        text = result["text"]
        if user_lang != "en":
            return await translate_text(text, user_lang)
        return text
    except Exception as e:
        return f"Transcription failed: {e}"

def retrieve_relevant_info(user_input, docs, faiss_index, k=3):
    user_emb = get_embedding(user_input)
    D, I = faiss_index.search(np.array([user_emb]), k)
    return [docs[idx] for idx in I[0] if 0 <= idx < len(docs)]

async def rag_response(transcribed_text, user_lang, selected_model):
    translated_input = await translate_text(transcribed_text, target_lang='en')
    relevant_info = retrieve_relevant_info(translated_input, rag_documents, index)
    rag_context = "\n\n".join(relevant_info)

    full_prompt = f"""Relevant excerpts:\n{rag_context}\n\nQuestion:\n{translated_input}\n\nPlease respond..."""

    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": full_prompt}]
    response_text = ""

    if selected_model == "ollama/llama3.2":
        stream = openai.chat.completions.create(model="llama3.2", messages=messages, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                response_text += delta
    else:
        github_client = OpenAI(base_url="https://models.github.ai/inference", api_key=os.getenv("GITHUB_TOKEN"))
        result = github_client.chat.completions.create(model=selected_model, messages=messages)
        response_text = result.choices[0].message.content

    translated_response = await translate_text(response_text, target_lang=user_lang)
    audio_path = f"output_{uuid.uuid4().hex}.mp3"
    gTTS(text=translated_response, lang=user_lang).save(audio_path)

    return translated_response, audio_path

async def clean_text_for_tts(text):
    return text.replace('*', '').replace('[', '').replace(']', '')

async def handle_text_input(text, lang, model):
    clean_text = await clean_text_for_tts(text)
    return await rag_response(clean_text, lang, model)

# ----------------- Gradio UI -----------------
with gr.Blocks() as demo:
    gr.Markdown("## 🎧 Whisper Medical Assistant")

    model_selector = gr.Dropdown(
        label="Select Model",
        choices=["ollama/llama3.2", "openai/gpt-4.1", "openai/gpt-4.1-mini"],
        value="ollama/llama3.2"
    )

    audio_input = gr.Audio(type="filepath", label="🎤 Speak")
    text_input = gr.Textbox(label="Or Type your question", placeholder="Type here...")
    lang_dropdown = gr.Dropdown(
        label="Select Language",
        choices=["en", "es", "fr", "de", "hi", "ta", "te", "ml", "kn"],
        value="en"
    )
    submit = gr.Button("Submit")

    transcribed_text = gr.Textbox(label="Transcription")
    response_output = gr.Textbox(label="Medical Advice")
    audio_output = gr.Audio(type="filepath", label="Audio Response")

    audio_input.change(transcribe_audio, inputs=[audio_input, lang_dropdown], outputs=[transcribed_text]).then(
        fn=rag_response,
        inputs=[transcribed_text, lang_dropdown, model_selector],
        outputs=[response_output, audio_output]
    )

    submit.click(fn=handle_text_input, inputs=[text_input, lang_dropdown, model_selector], outputs=[response_output, audio_output])

# ----------------- FastAPI App -----------------
app = FastAPI()

# Allow CORS (optional but helpful)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple test routes
@app.get("/base/hello")
def base_hello():
    return {"message": "Hello from Base"}

@app.get("/premium/hello")
def premium_hello():
    return {"message": "Hello from Premium"}

# Mount Gradio on `/`
@app.get("/", response_class=HTMLResponse)
def gradio_index():
    return demo.launch(share=False, inline=True)

