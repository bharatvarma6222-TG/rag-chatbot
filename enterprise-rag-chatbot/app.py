import os
from dotenv import load_dotenv
from openai import OpenAI
from utils.loader import load_pdf
from utils.chunker import chunk_text
from utils.vector_store import VectorStore

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text = load_pdf("data/sample.pdf")
chunks = chunk_text(text)

store = VectorStore()
store.build(chunks)

query = input("Ask a question: ")
context = "\n".join(store.search(query))

prompt = f"""
Answer using ONLY the context below.

Context:
{context}

Question:
{query}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2
)

print("\nAnswer:\n", response.choices[0].message.content)
