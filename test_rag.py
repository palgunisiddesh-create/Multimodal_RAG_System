import json
import faiss
import ollama
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIGURATION
# ============================================================

INDEX_PATH = "vectorstore/text_index.faiss"
METADATA_PATH = "vectorstore/text_metadata.json"

OLLAMA_MODEL = "llama3.2"
TOP_K = 5

# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("Loading FAISS index...")

try:
    index = faiss.read_index(INDEX_PATH)
except Exception as e:
    print("ERROR: Could not load FAISS index.")
    print(e)
    exit()

print("FAISS index loaded successfully.")

# ============================================================
# LOAD METADATA
# ============================================================

print("Loading metadata...")

try:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
except Exception as e:
    print("ERROR: Could not load metadata.")
    print(e)
    exit()

print(f"Metadata loaded successfully. Total items: {len(metadata)}")

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print("ERROR: Could not load embedding model.")
    print(e)
    exit()

print("Embedding model loaded successfully.")

# ============================================================
# ASK QUESTION
# ============================================================

question = input("\nEnter your question: ").strip()

if not question:
    print("Please enter a question.")
    exit()

# ============================================================
# CREATE QUESTION EMBEDDING
# ============================================================

print("\nCreating question embedding...")

question_embedding = embedding_model.encode(
    [question],
    convert_to_numpy=True
).astype("float32")

# ============================================================
# FAISS SEARCH
# ============================================================

print("Searching FAISS for relevant chunks...")

distances, indices = index.search(
    question_embedding,
    TOP_K
)

# ============================================================
# GET RETRIEVED CHUNKS
# ============================================================

retrieved_chunks = []

for position, idx in enumerate(indices[0]):

    if idx < 0:
        continue

    if idx >= len(metadata):
        continue

    item = metadata[idx]

    # Handle different metadata formats
    if isinstance(item, dict):

        text = (
            item.get("text")
            or item.get("content")
            or item.get("chunk")
            or item.get("page_content")
            or ""
        )

    elif isinstance(item, str):

        text = item

    else:

        text = str(item)

    text = str(text).strip()

    if text:

        retrieved_chunks.append({
            "rank": position + 1,
            "index": int(idx),
            "distance": float(distances[0][position]),
            "text": text
        })

# ============================================================
# DISPLAY RETRIEVED CHUNKS
# ============================================================

print("\n")
print("=" * 60)
print("              RETRIEVED FAISS CHUNKS")
print("=" * 60)

if not retrieved_chunks:

    print("\nNo chunks were retrieved from FAISS.")
    exit()

for item in retrieved_chunks:

    print(f"\n--- Chunk {item['rank']} ---")
    print(f"FAISS Index : {item['index']}")
    print(f"Distance    : {item['distance']:.4f}")
    print(f"Text        : {item['text'][:1000]}")

# ============================================================
# COMBINE CONTEXT
# ============================================================

context_parts = []

for item in retrieved_chunks:

    context_parts.append(
        f"""
CHUNK {item['rank']}:

{item['text']}
"""
    )

context = "\n".join(context_parts)

# ============================================================
# RAG PROMPT
# ============================================================

prompt = f"""
You are an intelligent document question-answering assistant.

You must answer the user's question using the retrieved document
information provided below.

================ RETRIEVED DOCUMENT =================

{context}

=======================================================

USER QUESTION:

{question}

=======================================================

IMPORTANT INSTRUCTIONS:

1. Read ALL retrieved chunks before answering.

2. Combine information from multiple chunks when necessary.

3. If the user asks about the main topic, overall topic, theme,
   subject, or what the document is about, identify the common
   subject across the retrieved chunks.

4. You are allowed to summarize information from the chunks.

5. Do NOT require the exact words from the question to appear
   in the document.

6. Do NOT invent information that is unrelated to the retrieved
   document.

7. If the retrieved chunks contain enough information to answer
   the question, ALWAYS provide an answer.

8. Only say "I could not find this information in the document"
   when the retrieved chunks genuinely contain no useful
   information for answering the question.

9. Give a clear and natural answer.

10. For a general topic question, answer in 2-3 sentences.

================ FINAL ANSWER =========================
"""

# ============================================================
# SEND TO OLLAMA
# ============================================================

print("\nGenerating answer using Ollama...")

try:

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful document analysis assistant. "
                    "Always use the supplied document context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"].strip()

except Exception as e:

    print("\nERROR: Could not communicate with Ollama.")
    print(e)
    exit()

# ============================================================
# DISPLAY FINAL ANSWER
# ============================================================

print("\n")
print("=" * 60)
print("                    RAG ANSWER")
print("=" * 60)

print("\n" + answer)

print("\n")
print("=" * 60)
print("                 RAG PIPELINE COMPLETE")
print("=" * 60)

print("\nPipeline used:")
print("Question")
print("   ↓")
print("Sentence Transformer")
print("   ↓")
print("Question Embedding")
print("   ↓")
print("FAISS Similarity Search")
print(f"   ↓")
print(f"Top {len(retrieved_chunks)} Relevant Chunks")
print("   ↓")
print("Ollama / Llama 3.2")
print("   ↓")
print("Final Answer")
print()