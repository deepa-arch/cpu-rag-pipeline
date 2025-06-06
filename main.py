from modules.pipeline import RAGPipeline

def load_documents(filepath="data/docs.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

documents = load_documents()
pipeline = RAGPipeline(documents)

while True:
    query = input("Ask a question (or type 'exit'): ")
    if query.lower() == 'exit':
        break
    answer = pipeline.query(query)
    print("\n🧠 Answer:\n", answer)
