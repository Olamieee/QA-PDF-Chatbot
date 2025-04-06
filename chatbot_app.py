from fastapi import FastAPI, File, UploadFile
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader
from io import BytesIO
import os

app = FastAPI()

HF_API_TOKEN = "your_huggingface_api_token" #HuggingFace API token

#Initialize HuggingFace model
hf = HuggingFaceHub(
    repo_id="facebook/llama-2-7b-hf",
    api_key=HF_API_TOKEN
)

#Initialize QA chain
qa_chain = load_qa_chain(hf)

#Endpoint to upload PDF
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_content = await file.read()
    file_path = f"uploaded_files/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(pdf_content)
    return {"message": "File uploaded successfully", "file_name": file.filename}

#Endpoint to ask questions from uploaded PDFs
@app.post("/ask/")
async def ask_question(file_name: str, question: str):
    try:
        with open(f"uploaded_files/{file_name}", "rb") as f:
            pdf_content = f.read()
        
        pdf_loader = PyPDFLoader(BytesIO(pdf_content))
        documents = pdf_loader.load()
        answer = qa_chain.run(input_documents=documents, question=question)
        return {"answer": answer}
    except FileNotFoundError:
        return {"error": "File not found"}

#Endpoint to ask a question based on a specific uploaded file
@app.post("/ask-from-file/")
async def ask_from_file(file_name: str, question: str):
    try:
        if file_name not in os.listdir("uploaded_files/"):
            return {"error": "File not found"}
        
        with open(f"uploaded_files/{file_name}", "rb") as f:
            pdf_content = f.read()
        
        pdf_loader = PyPDFLoader(BytesIO(pdf_content))
        documents = pdf_loader.load()
        answer = qa_chain.run(input_documents=documents, question=question)
        return {"answer": answer}
    except FileNotFoundError:
        return {"error": "File not found"}

#Endpoint to list uploaded PDFs
@app.get("/get-pdf-list/")
async def get_pdf_list():
    pdf_files = os.listdir("uploaded_files/")
    return {"pdf_files": pdf_files}

#Endpoint to check server health
@app.get("/health-check/")
async def health_check():
    return {"status": "OK"}