import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class NewsRecommender:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.articles = []

    def build_index(self, articles):
        self.articles = articles
        texts = [a["title"] + " " + (a.get("description") or "") for a in articles]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def recommend(self, query, top_k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        return [self.articles[i] for i in I[0]]
