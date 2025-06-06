# 🧠 CPU-Based RAG Inference Pipeline

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline fully optimized for **CPU-only environments**. It integrates document retrieval and large language model inference using lightweight components like `llama.cpp` and `sentence-transformers`.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/cpu-rag-pipeline.git
cd cpu-rag-pipeline
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Your Documents

Create a `docs.txt` file inside the `data/` folder:

```txt
Retrieval-Augmented Generation (RAG) is a technique that combines document retrieval with large language models for answering questions.
FastAPI is a modern Python web framework used for building APIs quickly.
LLaMA 2 is a family of open-weight language models developed by Meta.
FAISS is a library for efficient similarity search and clustering of dense vectors.
CTransformers is a Python library to run GGUF-based LLMs on CPU using llama.cpp backend.
```

---

## 📦 Model Download/Setup

### 1. Download GGUF Model (e.g., LLaMA 2)

* Go to: [https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
* Download: `llama-2-7b-chat.Q4_K_M.gguf`
* Place it in the `models/` directory:

```bash
models/llama-2-7b-chat.Q4_K_M.gguf
```

---

## 🚀 Example Usage

### CLI:

```bash
python main.py
```

**Input:** `What is RAG?`
**Output:** LLM-generated answer from retrieved document context.

### API:

```bash
uvicorn api.app:app --reload
```

Then open: [http://localhost:8000/docs](http://localhost:8000/docs)

**Sample Request:**

```json
POST /query
{
  "query": "What is FAISS used for?",
  "top_k": 3
}
```

---

## 📊 Performance Benchmarks (Approx.)

| Task                     | CPU Only (i5/8GB RAM) | Note                        |
| ------------------------ | --------------------- | --------------------------- |
| Embedding generation     | \~30ms per document   | Using MiniLM                |
| Similarity search        | <10ms                 | FAISS (FlatL2, CPU backend) |
| LLM inference (Q4\_K\_M) | \~2s per response     | llama.cpp via ctransformers |

---

## 🔧 Extending the Pipeline

### ✅ Switch Embedding Model:

Change this line in `retriever.py`:

```python
self.model = SentenceTransformer("all-MiniLM-L6-v2")
```

To any other:

```python
self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
```

### ✅ Switch LLM Model:

Download another `.gguf` file and update the path in `generator.py`:

```python
model_path = "models/your-model-name.gguf"
```

---

## 🧪 Test Questions You Can Ask:

* What is RAG?
* What is FastAPI used for?
* Who developed LLaMA 2?
* What is FAISS?
* What does CTransformers use under the hood?

---

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
├── main.py             # CLI entry point
├── requirements.txt
└── README.md
```

---

## ✅ License

MIT

---

> Built for low-resource environments. No GPU? No problem.
