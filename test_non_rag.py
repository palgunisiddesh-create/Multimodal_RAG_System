import ollama

document = """
Artificial Intelligence is a branch of computer science.
Machine learning allows computers to learn patterns from data.
Deep learning uses neural networks with multiple layers.
"""

question = "What is machine learning?"

prompt = f"""
Answer the question using the document below.

DOCUMENT:
{document}

QUESTION:
{question}

Give a clear and concise answer.
"""

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\n========== NON-RAG ANSWER ==========\n")
print(response["message"]["content"])