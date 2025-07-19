from fastapi import FastAPI, UploadFile, Form
from prescription import analyze_prescription
from basic import run_basic_model
from premium import run_premium_model
from rag import answer_with_rag




app = FastAPI()

@app.post("/prescription")
async def analyze(file: UploadFile, api_key: str = Form(...), image_type: str = Form("Prescription Receipt")):
    image_bytes = await file.read()
    result = analyze_prescription(image_bytes, api_key, image_type)
    return {"analysis": result}


@app.get("/basic")
def basic():
    return {"result": run_basic_model()}

@app.get("/premium")
def premium():
    return {"result": run_premium_model()}

@app.post("/rag")
async def rag(input_text: str = Form(...)):
    return {"answer": answer_with_rag(input_text)}
