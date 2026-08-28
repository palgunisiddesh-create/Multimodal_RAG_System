from utils.text_faiss import create_text_faiss
import os


pdf_path = "uploads/AI_RAG_Document.pdf"


if os.path.exists(pdf_path):

    create_text_faiss(pdf_path)

else:

    print("PDF not found:", pdf_path)