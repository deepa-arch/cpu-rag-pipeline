from sentence_transformers import SentenceTransformer
import pickle

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: list[str]):
        return self.model.encode(documents, show_progress_bar=True)

    def save_embeddings(self, embeddings, filepath="embeddings/doc_embeddings.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(embeddings, f)

    def load_embeddings(self, filepath="embeddings/doc_embeddings.pkl"):
        with open(filepath, "rb") as f:
            return pickle.load(f)
