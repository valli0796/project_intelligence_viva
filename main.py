import os
import uuid
from pathlib import Path
import chromadb
import pymupdf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(name="project_viva_documents")

app = FastAPI(title="Project Intelligence & AI Viva System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VivaQuestion(BaseModel):
    question: str

def split_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks

@app.get("/")
def home():
    return {"message": "Welcome to Project Intelligence & AI Viva System"}

@app.get("/health")
def health_check():
    return {"status": "running"}

@app.post("/upload-project")
async def upload_project(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF project report.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            text = "\n".join(page.get_text() for page in pdf)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the PDF file.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable text found. The PDF may be scanned.")

    safe_name = Path(file.filename).name
    saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    saved_path.write_bytes(content)
    chunks = split_text(text)
    embeddings = embedding_model.encode(chunks, normalize_embeddings=True).tolist()

    global collection
    chroma_client.delete_collection("project_viva_documents")
    collection = chroma_client.create_collection(name="project_viva_documents")
    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"filename": safe_name, "chunk": index} for index in range(len(chunks))]
    )

    return {
        "message": "Project uploaded and indexed successfully.",
        "filename": safe_name,
        "total_chunks": len(chunks)
    }

@app.post("/ask-viva")
def ask_viva(request: VivaQuestion):
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="Upload a project PDF before starting the viva.")

    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing from the backend .env file.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    query_embedding = embedding_model.encode(question, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=min(4, collection.count()))
    documents = results.get("documents", [[]])[0]
    context = "\n\n".join(documents)

    try:
        client = genai.Client(api_key=API_KEY)
        prompt = f"""
You are the AI viva examiner for a student's academic project.

Answer the student's question using the supplied project context.
If the context does not contain enough information, clearly say so.
Do not invent project-specific facts.
Explain answers in simple language suitable for a student.

Project context:
{context}

Student question:
{question}
"""
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        answer = response.text

        if not answer:
            raise ValueError("The AI returned an empty response.")

        return {"question": question, "answer": answer, "sources": documents}

    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI response failed. Check your API key and model access. {exc}")