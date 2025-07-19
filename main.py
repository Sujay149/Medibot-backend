from fastapi import FastAPI, UploadFile, Form
from prescription import analyze_prescription
from basic import run_basic_model
from premium import run_premium_model
from rag import answer_with_rag

app = FastAPI()

@app.post("/prescription")
async def prescription(file: UploadFile):
    contents = await file.read()
    result = analyze_prescription(contents)
    return {"result": result}

@app.get("/basic")
def basic():
    return {"result": run_basic_model()}

@app.get("/premium")
def premium():
    return {"result": run_premium_model()}

@app.post("/rag")
async def rag(input_text: str = Form(...)):
    return {"answer": answer_with_rag(input_text)}
