from fastapi import FastAPI
from pydantic import BaseModel
from modules.pipeline import RAGPipeline

# --------- Load documents from file ---------
def load_documents(path="data/docs.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# --------- Instantiate RAG Pipeline once ---------
documents = load_documents()
pipeline = RAGPipeline(documents)

# --------- Setup FastAPI ---------
app = FastAPI(title="CPU-Based RAG Inference API")

# --------- Request & Response Schemas ---------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str

# --------- Routes ---------
@app.get("/")
def health_check():
    return {"message": "RAG API is running."}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    answer = pipeline.query(request.query, top_k=request.top_k)
    return QueryResponse(answer=answer)
