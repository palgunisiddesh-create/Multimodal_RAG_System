import streamlit as st
import os
import time
import tempfile
import fitz
import pdfplumber
import faiss
import numpy as np
import ollama

from PIL import Image
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import BlipProcessor, BlipForConditionalGeneration


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multimodal RAG Comparison",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Multimodal RAG Comparison Dashboard")

st.write(
    "Compare Non-RAG and Multimodal RAG using Ollama, "
    "FAISS, Sentence Transformers and BLIP."
)

st.divider()


# ============================================================
# LOAD TEXT EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# LOAD BLIP
# ============================================================

@st.cache_resource
def load_blip():

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return processor, model


# ============================================================
# EXTRACT TEXT
# ============================================================

def extract_text(uploaded_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp:

        temp.write(
            uploaded_file.getbuffer()
        )

        pdf_path = temp.name

    text = ""

    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                page_text = page.extract_text()

                if page_text:

                    text += (
                        f"\n--- Page {page_number} ---\n"
                        f"{page_text}\n"
                    )

    finally:

        os.remove(pdf_path)

    return text


# ============================================================
# EXTRACT IMAGES
# ============================================================

def extract_images(uploaded_file):

    temp_dir = tempfile.mkdtemp()

    pdf_path = os.path.join(
        temp_dir,
        "document.pdf"
    )

    with open(pdf_path, "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )

    doc = fitz.open(pdf_path)

    image_paths = []

    image_number = 0

    for page_number in range(
        len(doc)
    ):

        page = doc[page_number]

        images = page.get_images(
            full=True
        )

        for img in images:

            xref = img[0]

            image_data = doc.extract_image(
                xref
            )

            image_bytes = image_data[
                "image"
            ]

            extension = image_data[
                "ext"
            ]

            image_number += 1

            image_path = os.path.join(
                temp_dir,
                f"image_{image_number}.{extension}"
            )

            with open(
                image_path,
                "wb"
            ) as f:

                f.write(image_bytes)

            image_paths.append(
                image_path
            )

    doc.close()

    return image_paths


# ============================================================
# CREATE TEXT CHUNKS
# ============================================================

def create_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_text(text)


# ============================================================
# CREATE TEXT FAISS
# ============================================================

def create_text_faiss(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


# ============================================================
# IMAGE CAPTIONING
# ============================================================

def generate_image_captions(image_paths):

    processor, model = load_blip()

    results = []

    for image_path in image_paths:

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            output = model.generate(
                **inputs,
                max_new_tokens=50
            )

            caption = processor.decode(
                output[0],
                skip_special_tokens=True
            )

            results.append({
                "path": image_path,
                "caption": caption
            })

        except Exception as e:

            results.append({
                "path": image_path,
                "caption": f"Error: {e}"
            })

    return results


# ============================================================
# CREATE IMAGE FAISS
# ============================================================

def create_image_faiss(image_results):

    captions = [
        item["caption"]
        for item in image_results
    ]

    if not captions:

        return None

    embeddings = embedding_model.encode(
        captions,
        convert_to_numpy=True
    ).astype("float32")

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


# ============================================================
# RETRIEVE TEXT
# ============================================================

def retrieve_text(
    question,
    index,
    chunks,
    top_k=5
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    k = min(
        top_k,
        len(chunks)
    )

    distances, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for rank, idx in enumerate(
        indices[0]
    ):

        if idx >= 0:

            results.append({
                "rank": rank + 1,
                "distance": float(
                    distances[0][rank]
                ),
                "text": chunks[idx]
            })

    return results


# ============================================================
# RETRIEVE IMAGES
# ============================================================

def retrieve_images(
    question,
    image_index,
    image_results,
    top_k=3
):

    if image_index is None:

        return []

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    k = min(
        top_k,
        len(image_results)
    )

    distances, indices = image_index.search(
        question_embedding,
        k
    )

    results = []

    for rank, idx in enumerate(
        indices[0]
    ):

        if idx >= 0:

            item = image_results[idx]

            results.append({
                "rank": rank + 1,
                "distance": float(
                    distances[0][rank]
                ),
                "path": item["path"],
                "caption": item["caption"]
            })

    return results


# ============================================================
# OLLAMA
# ============================================================

def ask_ollama(prompt):

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response[
        "message"
    ][
        "content"
    ]


# ============================================================
# NON-RAG
# ============================================================

def non_rag_answer(
    document_text,
    question
):

    prompt = f"""
You are a document analysis assistant.

DOCUMENT:
{document_text}

QUESTION:
{question}

Answer the question using the document.

If the question asks for the main topic or overall theme,
summarize the common subject of the document.

Do not invent information.

ANSWER:
"""

    return ask_ollama(prompt)


# ============================================================
# MULTIMODAL RAG
# ============================================================

def multimodal_rag_answer(
    question,
    text_results,
    image_results
):

    text_context = "\n\n".join(
        [
            f"TEXT CHUNK {x['rank']}:\n{x['text']}"
            for x in text_results
        ]
    )

    image_context = "\n\n".join(
        [
            f"IMAGE {x['rank']}:\n"
            f"Caption: {x['caption']}"
            for x in image_results
        ]
    )

    prompt = f"""
You are a multimodal document question-answering assistant.

Use the retrieved text and image information to answer the question.

================ TEXT =================

{text_context}

================ IMAGES ===============

{image_context}

========================================

QUESTION:
{question}

Instructions:

1. Use the retrieved text and image captions.
2. Combine information when appropriate.
3. If the question asks for the main topic,
   identify the common theme.
4. Do not invent unsupported information.
5. Give a clear answer.

FINAL ANSWER:
"""

    return ask_ollama(prompt)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.success(
        "Ollama: Llama 3.2"
    )

    st.info(
        "Text: FAISS"
    )

    st.info(
        "Images: BLIP + FAISS"
    )

    st.info(
        "Embeddings: MiniLM"
    )


# ============================================================
# UPLOAD
# ============================================================

st.header("📄 1. Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS
# ============================================================

if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    with st.spinner(
        "Extracting PDF text..."
    ):

        document_text = extract_text(
            uploaded_file
        )

    # --------------------------------------------------------
    # CHUNKS
    # --------------------------------------------------------

    chunks = create_chunks(
        document_text
    )

    # --------------------------------------------------------
    # TEXT FAISS
    # --------------------------------------------------------

    with st.spinner(
        "Creating text FAISS index..."
    ):

        text_index = create_text_faiss(
            chunks
        )

    # --------------------------------------------------------
    # IMAGES
    # --------------------------------------------------------

    with st.spinner(
        "Extracting images..."
    ):

        image_paths = extract_images(
            uploaded_file
        )

    # --------------------------------------------------------
    # BLIP
    # --------------------------------------------------------

    image_results = []

    image_index = None

    if image_paths:

        with st.spinner(
            "Generating BLIP image captions..."
        ):

            image_results = generate_image_captions(
                image_paths
            )

        with st.spinner(
            "Creating image FAISS index..."
        ):

            image_index = create_image_faiss(
                image_results
            )

    # --------------------------------------------------------
    # DOCUMENT METRICS
    # --------------------------------------------------------

    st.header("📊 2. Document Information")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Characters",
            len(document_text)
        )

    with c2:

        st.metric(
            "Text Chunks",
            len(chunks)
        )

    with c3:

        st.metric(
            "Images",
            len(image_results)
        )

    with c4:

        st.metric(
            "FAISS Text Vectors",
            text_index.ntotal
        )

    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    if image_results:

        st.header(
            "🖼️ 3. Extracted Images"
        )

        image_columns = st.columns(
            min(3, len(image_results))
        )

        for i, item in enumerate(
            image_results
        ):

            with image_columns[
                i % len(image_columns)
            ]:

                st.image(
                    item["path"],
                    use_container_width=True
                )

                st.caption(
                    item["caption"]
                )

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    st.header(
        "❓ 4. Ask Question"
    )

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: What is the main topic "
            "of the document?"
        )
    )

    ask_button = st.button(
        "🚀 ASK QUESTION",
        use_container_width=True
    )

    # ========================================================
    # ANSWERS
    # ========================================================

    if ask_button and question:

        # ----------------------------------------------------
        # NON-RAG
        # ----------------------------------------------------

        with st.spinner(
            "Running Non-RAG..."
        ):

            start = time.time()

            answer1 = non_rag_answer(
                document_text,
                question
            )

            non_rag_time = (
                time.time() - start
            )

        # ----------------------------------------------------
        # TEXT RETRIEVAL
        # ----------------------------------------------------

        with st.spinner(
            "Retrieving text using FAISS..."
        ):

            start = time.time()

            text_results = retrieve_text(
                question,
                text_index,
                chunks,
                5
            )

            text_retrieval_time = (
                time.time() - start
            )

        # ----------------------------------------------------
        # IMAGE RETRIEVAL
        # ----------------------------------------------------

        with st.spinner(
            "Retrieving relevant images..."
        ):

            start = time.time()

            image_results_retrieved = retrieve_images(
                question,
                image_index,
                image_results,
                3
            )

            image_retrieval_time = (
                time.time() - start
            )

        # ----------------------------------------------------
        # MULTIMODAL RAG
        # ----------------------------------------------------

        with st.spinner(
            "Generating Multimodal RAG answer..."
        ):

            start = time.time()

            answer2 = multimodal_rag_answer(
                question,
                text_results,
                image_results_retrieved
            )

            rag_generation_time = (
                time.time() - start
            )

        rag_time = (
            text_retrieval_time
            + image_retrieval_time
            + rag_generation_time
        )

        # ====================================================
        # ANSWER COMPARISON
        # ====================================================

        st.header(
            "🔍 5. Answer Comparison"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "🔵 WITHOUT RAG"
            )

            st.write(
                answer1
            )

            st.metric(
                "Response Time",
                f"{non_rag_time:.2f} sec"
            )

            st.info(
                "Direct document → Ollama"
            )

        with col2:

            st.subheader(
                "🟢 MULTIMODAL RAG"
            )

            st.write(
                answer2
            )

            st.metric(
                "Response Time",
                f"{rag_time:.2f} sec"
            )

            st.success(
                "Text FAISS + Image FAISS + Ollama"
            )

        # ====================================================
        # RETRIEVED TEXT
        # ====================================================

        st.header(
            "📝 6. Retrieved Text Chunks"
        )

        for item in text_results:

            with st.expander(
                f"Chunk {item['rank']} "
                f"| Distance: {item['distance']:.4f}"
            ):

                st.write(
                    item["text"]
                )

        # ====================================================
        # RETRIEVED IMAGES
        # ====================================================

        st.header(
            "🖼️ 7. Retrieved Images"
        )

        if image_results_retrieved:

            image_cols = st.columns(
                min(
                    3,
                    len(
                        image_results_retrieved
                    )
                )
            )

            for i, item in enumerate(
                image_results_retrieved
            ):

                with image_cols[
                    i % len(image_cols)
                ]:

                    st.image(
                        item["path"],
                        use_container_width=True
                    )

                    st.caption(
                        f"Image {item['rank']}: "
                        f"{item['caption']}"
                    )

        else:

            st.info(
                "No images were retrieved."
            )

        # ====================================================
        # PERFORMANCE
        # ====================================================

        st.header(
            "📊 8. Performance Comparison"
        )

        p1, p2, p3, p4 = st.columns(4)

        with p1:

            st.metric(
                "Non-RAG",
                f"{non_rag_time:.2f}s"
            )

        with p2:

            st.metric(
                "Multimodal RAG",
                f"{rag_time:.2f}s"
            )

        with p3:

            st.metric(
                "Text Retrieved",
                len(text_results)
            )

        with p4:

            st.metric(
                "Images Retrieved",
                len(image_results_retrieved)
            )

        # ====================================================
        # ARCHITECTURE
        # ====================================================

        st.header(
            "🏗️ 9. Pipeline Architecture"
        )

        a1, a2 = st.columns(2)

        with a1:

            st.subheader(
                "Without RAG"
            )

            st.code(
                """
PDF
 ↓
Text Extraction
 ↓
Full Document
 ↓
Ollama
 ↓
Answer
                """
            )

        with a2:

            st.subheader(
                "Multimodal RAG"
            )

            st.code(
                """
PDF
 ↓
 ├── Text → Chunks → Embeddings → FAISS
 │
 └── Images → BLIP → Captions → Embeddings → FAISS
                         ↓
                  Relevant Context
                         ↓
                       Ollama
                         ↓
                       Answer
                """
            )

else:

    st.info(
        "Upload a PDF to start."
    ) 