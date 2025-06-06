# 🧠 CPU-Based RAG Inference Pipeline

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline fully optimized for **CPU-only environments**. It integrates document retrieval and large language model inference using lightweight components like `llama.cpp` and `sentence-transformers`.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/cpu-rag-pipeline.git
cd cpu-rag-pipeline
2. Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Prepare Your Documents
Create a docs.txt file inside the data/ folder:

txt
Copy
Edit
Retrieval-Augmented Generation (RAG) is a technique that combines document retrieval with large language models for answering questions.
FastAPI is a modern Python web framework used for building APIs quickly.
LLaMA 2 is a family of open-weight language models developed by Meta.
FAISS is a library for efficient similarity search and clustering of dense vectors.
CTransformers is a Python library to run GGUF-based LLMs on CPU using llama.cpp backend.
📦 Model Download (Required Before Use)
⚠️ Important: The model file is large and has been excluded from the GitHub repository using .gitignore.

To run this project, download the required .gguf model file manually:

Visit: TheBloke/Llama-2-7B-Chat-GGUF on Hugging Face

Download this file:

Copy
Edit
llama-2-7b-chat.Q4_K_M.gguf
Create a models/ folder (if not already present) and place the file inside it:

bash
Copy
Edit
cpu-rag-pipeline/models/llama-2-7b-chat.Q4_K_M.gguf
🚀 Example Usage
CLI:
bash
Copy
Edit
python main.py
Input: What is RAG?
Output: LLM-generated answer from retrieved document context.

API:
bash
Copy
Edit
uvicorn api.app:app --reload
Then open: http://localhost:8000/docs

Sample Request:

json
Copy
Edit
POST /query
{
  "query": "What is FAISS used for?",
  "top_k": 3
}
📊 Performance Benchmarks (Approx.)
Task	CPU Only (i5/8GB RAM)	Note
Embedding generation	~30ms per document	Using MiniLM
Similarity search	<10ms	FAISS (FlatL2, CPU backend)
LLM inference (Q4_K_M)	~2s per response	llama.cpp via ctransformers

🔧 Extending the Pipeline
✅ Switch Embedding Model:
In retriever.py:

python
Copy
Edit
self.model = SentenceTransformer("all-MiniLM-L6-v2")
Change it to any other supported model:

python
Copy
Edit
self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
✅ Switch LLM Model:
Download a different .gguf model and update the path in generator.py:

python
Copy
Edit
model_path = "models/your-model-name.gguf"
🧪 Test Questions You Can Ask:
What is RAG?

What is FastAPI used for?

Who developed LLaMA 2?

What is FAISS?

What does CTransformers use under the hood?

📁 Project Structure
bash
Copy
Edit
cpu-rag-pipeline/
├── api/                # FastAPI app
│   └── app.py
├── data/               # Contains docs.txt
├── embeddings/         # (Optional) stores .npy files
├── models/             # Place your .gguf models here
├── modules/            # Pipeline logic
│   ├── generator.py
│   ├── retriever.py
│   └── pipeline.py
├── main.py             # CLI entry point
├── requirements.txt
└── README.md
✅ License
MIT