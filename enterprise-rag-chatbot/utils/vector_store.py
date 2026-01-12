import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def build(self, chunks):
        embeddings = self.embedder.encode(chunks)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))
        self.chunks = chunks

    def search(self, query, k=3):
        q_emb = self.embedder.encode([query])
        _, idx = self.index.search(np.array(q_emb), k)
        return [self.chunks[i] for i in idx[0]]
