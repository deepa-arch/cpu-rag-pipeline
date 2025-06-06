from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Retriever:
    def __init__(self, documents, model_name="all-MiniLM-L6-v2"):
        self.documents = documents
        self.embedding_model = SentenceTransformer(model_name)

        # ✅ Get embeddings
        self.embeddings = self.embedding_model.encode(documents)

        # ✅ Fix: Get the dimension of each embedding
        dimension = self.embeddings.shape[1]

        # ✅ Build FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]
