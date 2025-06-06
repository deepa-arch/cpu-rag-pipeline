# 🧠 CPU-Based RAG Inference Pipeline

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline fully optimized for **CPU-only environments**. It integrates document retrieval and large language model inference using lightweight components like `llama.cpp` and `sentence-transformers`.

## 📋 Table of Contents
- [Setup Instructions](#️-setup-instructions)
- [Model Download](#-model-download)
- [Usage Examples](#-example-usage)
- [Performance Benchmarks](#-performance-benchmarks)
- [Extending the Pipeline](#-extending-the-pipeline)
- [Test Questions](#-test-questions)
- [Project Structure](#-project-structure)
- [License](#-license)

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/deepa-arch/cpu-rag-pipeline.git
cd cpu-rag-pipeline
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Your Documents
Create a `docs.txt` file inside the `data/` folder:
```text
Retrieval-Augmented Generation (RAG) is a technique that combines document retrieval with large language models for answering questions.
FastAPI is a modern Python web framework used for building APIs quickly.
LLaMA 2 is a family of open-weight language models developed by Meta.
FAISS is a library for efficient similarity search and clustering of dense vectors.
CTransformers is a Python library to run GGUF-based LLMs on CPU using llama.cpp backend.
```

## 📦 Model Download

⚠️ **Important**:The model file is large and has been excluded from the GitHub repository using .gitignore.

To run this project, download the required .gguf model file manually:

1. Visit [TheBloke/Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) on Hugging Face
2. Download: `llama-2-7b-chat.Q4_K_M.gguf`
3. Place in models folder:
```bash
cpu-rag-pipeline/models/llama-2-7b-chat.Q4_K_M.gguf
```

## 🚀 Example Usage

### CLI Interface
```bash
python main.py
# Input: What is RAG?
# Output: LLM-generated answer from retrieved document context.
```

### API Interface
```bash
uvicorn api.app:app --reload
# Then open: http://localhost:8000/docs
```

Sample API Request:
```json
POST /query
{
  "query": "What is FAISS used for?",
  "top_k": 3
}
```

## 📊 Performance Benchmarks

| Task | CPU Only (i5/8GB RAM) | Note |
|------|----------------------|------|
| Embedding generation | ~30ms per document | Using MiniLM |
| Similarity search | <10ms | FAISS (FlatL2, CPU backend) |
| LLM inference (Q4_K_M) | ~2s per response | llama.cpp via ctransformers |

## 🔧 Extending the Pipeline

### Switch Embedding Model
```python
# In retriever.py
self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
```

### Switch LLM Model
```python
# In generator.py
model_path = "models/your-model-name.gguf"
```

## 🧪 Test Questions
- What is RAG?
- What is FastAPI used for?
- Who developed LLaMA 2?
- What is FAISS?
- What does CTransformers use under the hood?

## 📁 Project Structure
```
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
├── main.py            # CLI entry point
├── requirements.txt
└── README.md
```

## 📄 License
MIT