from ctransformers import AutoModelForCausalLM

class Generator:
    def __init__(self, model_path="models/llama-2-7b-chat.Q4_K_M.gguf", max_tokens=256):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="llama",
            max_new_tokens=max_tokens,
            threads=4  # Use more if your CPU has more cores
        )

    def generate(self, context, query):
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        return self.model(prompt)
