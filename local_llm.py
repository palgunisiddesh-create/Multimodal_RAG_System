import ollama


def generate_answer(question, text_results, image_results):

    text_context = "\n".join(text_results)
    image_context = "\n".join(image_results)

    prompt = f"""
You are a helpful Multimodal RAG assistant.

Answer the question using the retrieved information.

Question:
{question}

Retrieved Text:
{text_context}

Retrieved Image Information:
{image_context}

Give a clear and simple answer.
Do not make up information that is not present in the retrieved context.
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]