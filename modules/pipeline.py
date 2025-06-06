from modules.retriever import Retriever
from modules.generator import Generator

class RAGPipeline:
    def __init__(self, documents):
        self.retriever = Retriever(documents)
        self.generator = Generator()

    def query(self, user_question, top_k=3):
        relevant_chunks = self.retriever.retrieve(user_question, top_k=top_k)
        context = "\n".join(relevant_chunks)
        return self.generator.generate(context, user_question)
