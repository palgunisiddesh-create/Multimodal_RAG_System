import os
import json
import shutil
import requests
import numpy as np
import streamlit as st
import faiss
import fitz
import pdfplumber
import pytesseract

from PIL import Image
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import BlipProcessor, BlipForConditionalGeneration


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MAX_PDF_SIZE = 700 * 1024 * 1024

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

VECTORSTORE_DIR = os.path.join(
    BASE_DIR,
    "vectorstore"
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "extracted_images"
)

OCR_DIR = os.path.join(
    BASE_DIR,
    "ocr_pages"
)

TEXT_INDEX_PATH = os.path.join(
    VECTORSTORE_DIR,
    "text_index.faiss"
)

TEXT_METADATA_PATH = os.path.join(
    VECTORSTORE_DIR,
    "text_metadata.json"
)

IMAGE_INDEX_PATH = os.path.join(
    VECTORSTORE_DIR,
    "image_index.faiss"
)

IMAGE_METADATA_PATH = os.path.join(
    VECTORSTORE_DIR,
    "image_metadata.json"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OCR_DIR, exist_ok=True)


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "processed_pdf" not in st.session_state:
    st.session_state.processed_pdf = ""

if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = 0

if "ocr_used" not in st.session_state:
    st.session_state.ocr_used = False

if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


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
# LOAD TEXT FAISS
# ============================================================

def load_text_database():

    if not os.path.exists(TEXT_INDEX_PATH):
        return None, []

    if not os.path.exists(TEXT_METADATA_PATH):
        return None, []

    try:

        index = faiss.read_index(
            TEXT_INDEX_PATH
        )

        with open(
            TEXT_METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

        return index, metadata

    except Exception as e:

        st.warning(
            f"Could not load text database: {e}"
        )

        return None, []


# ============================================================
# LOAD IMAGE FAISS
# ============================================================

def load_image_database():

    if not os.path.exists(IMAGE_INDEX_PATH):
        return None, []

    if not os.path.exists(IMAGE_METADATA_PATH):
        return None, []

    try:

        index = faiss.read_index(
            IMAGE_INDEX_PATH
        )

        with open(
            IMAGE_METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

        return index, metadata

    except Exception as e:

        st.warning(
            f"Could not load image database: {e}"
        )

        return None, []


# ============================================================
# DELETE DATABASE
# ============================================================

def clear_database():

    files = [
        TEXT_INDEX_PATH,
        TEXT_METADATA_PATH,
        IMAGE_INDEX_PATH,
        IMAGE_METADATA_PATH
    ]

    for file_path in files:

        if os.path.exists(file_path):

            try:
                os.remove(file_path)
            except Exception:
                pass

    if os.path.exists(IMAGE_DIR):

        for item in os.listdir(IMAGE_DIR):

            path = os.path.join(
                IMAGE_DIR,
                item
            )

            try:

                if os.path.isfile(path):
                    os.remove(path)

                elif os.path.isdir(path):
                    shutil.rmtree(path)

            except Exception:
                pass

    if os.path.exists(OCR_DIR):

        for item in os.listdir(OCR_DIR):

            path = os.path.join(
                OCR_DIR,
                item
            )

            try:

                if os.path.isfile(path):
                    os.remove(path)

            except Exception:
                pass


# ============================================================
# MODELS
# ============================================================

embedding_model = load_embedding_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Dashboard")

    st.success("📄 PDF Limit: 700 MB")
    st.success("🔎 FAISS Retrieval")
    st.success("🖼️ BLIP Images")
    st.success("🔤 Tesseract OCR")
    st.success(f"🤖 Ollama: {OLLAMA_MODEL}")

    st.divider()

    text_index_sidebar, text_metadata_sidebar = (
        load_text_database()
    )

    image_index_sidebar, image_metadata_sidebar = (
        load_image_database()
    )

    text_count = (
        text_index_sidebar.ntotal
        if text_index_sidebar is not None
        else 0
    )

    image_count = (
        image_index_sidebar.ntotal
        if image_index_sidebar is not None
        else 0
    )

    st.metric(
        "Text Chunks",
        text_count
    )

    st.metric(
        "Images",
        image_count
    )

    st.metric(
        "Questions",
        len(st.session_state.history)
    )

    if st.session_state.ocr_used:

        st.success("🔤 OCR Used")

    else:

        st.info("🔤 OCR Not Used")

    st.divider()

    st.subheader("🗑️ Database Management")

    if st.button(
        "Clear FAISS Database",
        use_container_width=True
    ):

        clear_database()

        st.session_state.processed_pdf = ""
        st.session_state.pdf_pages = 0
        st.session_state.ocr_used = False
        st.session_state.processing_complete = False

        st.success(
            "Database cleared."
        )

        st.rerun()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🤖 Multimodal RAG System"
)

st.markdown(
    """
    **PDF → Text + OCR + Images → FAISS → Retrieval → Ollama**
    """
)


# ============================================================
# PIPELINE
# ============================================================

st.header("🔄 RAG Pipeline")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.info("📄 PDF\n\nUpload")

with c2:
    st.info("🔤 OCR\n\nExtract")

with c3:
    st.info("🧩 FAISS\n\nStore")

with c4:
    st.info("🔎 RAG\n\nRetrieve")

with c5:
    st.info("🤖 Ollama\n\nAnswer")


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("📄 1. Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    help="Maximum allowed PDF size: 700 MB"
)


# ============================================================
# UPLOAD INFORMATION
# ============================================================

if uploaded_file is not None:

    file_size = uploaded_file.size

    size_mb = (
        file_size / (1024 * 1024)
    )

    st.info(
        f"📄 {uploaded_file.name} | "
        f"{size_mb:.2f} MB / 700 MB"
    )

    if file_size > MAX_PDF_SIZE:

        st.error(
            "❌ File is larger than 700 MB."
        )

        st.stop()

    st.success(
        "✅ PDF accepted."
    )

    # ========================================================
    # PROCESS PDF BUTTON
    # ========================================================

    if st.button(
        "🚀 Process PDF",
        type="primary",
        use_container_width=True
    ):

        safe_filename = os.path.basename(
            uploaded_file.name
        )

        pdf_path = os.path.join(
            UPLOAD_DIR,
            safe_filename
        )

        # ====================================================
        # SAVE PDF
        # ====================================================

        try:

            with open(
                pdf_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            st.success(
                "✅ PDF uploaded and saved."
            )

        except Exception as e:

            st.error(
                f"❌ Error saving PDF: {e}"
            )

            st.stop()


        # ====================================================
        # OPEN PDF
        # ====================================================

        try:

            pdf_document = fitz.open(
                pdf_path
            )

            total_pages = len(
                pdf_document
            )

            pdf_document.close()

            st.session_state.pdf_pages = (
                total_pages
            )

            st.session_state.processed_pdf = (
                safe_filename
            )

            st.info(
                f"📑 PDF contains "
                f"{total_pages} pages."
            )

        except Exception as e:

            st.error(
                f"❌ Cannot open PDF: {e}"
            )

            st.stop()


        # ====================================================
        # STEP 2: TEXT EXTRACTION
        # ====================================================

        st.header(
            "📖 2. Extracting PDF Text"
        )

        page_texts = []

        try:

            with pdfplumber.open(
                pdf_path
            ) as pdf:

                total = len(
                    pdf.pages
                )

                progress = st.progress(0)

                for page_number, page in enumerate(
                    pdf.pages
                ):

                    try:

                        text = page.extract_text()

                    except Exception:

                        text = None

                    if text and text.strip():

                        page_texts.append(
                            {
                                "page": page_number + 1,
                                "text": text,
                                "source": "PDF Text"
                            }
                        )

                    progress.progress(
                        (page_number + 1) / total
                    )

                progress.empty()

        except Exception as e:

            st.error(
                f"Text extraction error: {e}"
            )


        extracted_characters = sum(
            len(x["text"])
            for x in page_texts
        )


        # ====================================================
        # STEP 3: OCR
        # ====================================================

        if extracted_characters < 100:

            st.warning(
                """
                ⚠️ Very little normal text was found.

                This appears to be a scanned PDF.
                Tesseract OCR will now process the pages.
                """
            )

            if not os.path.exists(
                TESSERACT_PATH
            ):

                st.error(
                    "❌ Tesseract was not found at:\n"
                    f"{TESSERACT_PATH}"
                )

                st.stop()

            st.session_state.ocr_used = True

            ocr_results = []

            try:

                pdf_document = fitz.open(
                    pdf_path
                )

                total = len(
                    pdf_document
                )

                progress = st.progress(0)

                for page_number in range(total):

                    page = pdf_document[
                        page_number
                    ]

                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(
                            1.5,
                            1.5
                        )
                    )

                    image = Image.frombytes(
                        "RGB",
                        [
                            pixmap.width,
                            pixmap.height
                        ],
                        pixmap.samples
                    )

                    text = pytesseract.image_to_string(
                        image,
                        lang="eng"
                    )

                    if text and text.strip():

                        ocr_results.append(
                            {
                                "page": page_number + 1,
                                "text": text,
                                "source": "Tesseract OCR"
                            }
                        )

                        ocr_file = os.path.join(
                            OCR_DIR,
                            f"page_{page_number + 1}.txt"
                        )

                        with open(
                            ocr_file,
                            "w",
                            encoding="utf-8"
                        ) as f:

                            f.write(text)

                    progress.progress(
                        (page_number + 1) / total
                    )

                progress.empty()

                pdf_document.close()

                page_texts = ocr_results

                ocr_characters = sum(
                    len(x["text"])
                    for x in page_texts
                )

                st.success(
                    f"✅ OCR extracted "
                    f"{ocr_characters:,} characters."
                )

            except Exception as e:

                st.error(
                    f"❌ OCR error: {e}"
                )

        else:

            st.success(
                f"✅ Extracted "
                f"{extracted_characters:,} characters "
                f"from PDF text."
            )


        # ====================================================
        # STEP 4: CHUNKING
        # ====================================================

        st.header(
            "🧩 3. Creating Text Chunks"
        )

        chunks = []

        if page_texts:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            for page_item in page_texts:

                pieces = splitter.split_text(
                    page_item["text"]
                )

                for piece in pieces:

                    if piece.strip():

                        chunks.append(
                            {
                                "text": piece,
                                "page": page_item["page"],
                                "source": page_item["source"]
                            }
                        )

        st.success(
            f"🧩 Created "
            f"{len(chunks)} chunks."
        )


        # ====================================================
        # STEP 5: TEXT FAISS
        # ====================================================

        if chunks:

            st.header(
                "🔎 4. Creating Text FAISS"
            )

            with st.spinner(
                "Creating text embeddings..."
            ):

                chunk_texts = [
                    x["text"]
                    for x in chunks
                ]

                embeddings = (
                    embedding_model.encode(
                        chunk_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    ).astype("float32")
                )

            dimension = embeddings.shape[1]

            text_faiss = faiss.IndexFlatIP(
                dimension
            )

            text_faiss.add(
                embeddings
            )

            faiss.write_index(
                text_faiss,
                TEXT_INDEX_PATH
            )

            text_metadata = []

            for index_number, item in enumerate(
                chunks
            ):

                text_metadata.append(
                    {
                        "id": index_number,
                        "text": item["text"],
                        "page": item["page"],
                        "source": item["source"]
                    }
                )

            with open(
                TEXT_METADATA_PATH,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    text_metadata,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            st.success(
                f"✅ Text FAISS created with "
                f"{len(chunks)} vectors."
            )

        else:

            st.warning(
                "⚠️ No text chunks were created."
            )


        # ====================================================
        # STEP 6: IMAGE EXTRACTION
        # ====================================================

        st.header(
            "🖼️ 5. Extracting PDF Images"
        )

        extracted_images = []

        try:

            pdf_document = fitz.open(
                pdf_path
            )

            total = len(
                pdf_document
            )

            progress = st.progress(0)

            for page_number in range(total):

                page = pdf_document[
                    page_number
                ]

                images = page.get_images(
                    full=True
                )

                for image_number, image_info in enumerate(
                    images
                ):

                    try:

                        xref = image_info[0]

                        image_data = (
                            pdf_document.extract_image(
                                xref
                            )
                        )

                        image_bytes = (
                            image_data["image"]
                        )

                        extension = (
                            image_data["ext"]
                        )

                        filename = (
                            f"page_"
                            f"{page_number + 1}_"
                            f"image_"
                            f"{image_number + 1}."
                            f"{extension}"
                        )

                        image_path = os.path.join(
                            IMAGE_DIR,
                            filename
                        )

                        with open(
                            image_path,
                            "wb"
                        ) as f:

                            f.write(
                                image_bytes
                            )

                        extracted_images.append(
                            {
                                "path": image_path,
                                "filename": filename,
                                "page": page_number + 1
                            }
                        )

                    except Exception:
                        continue

                progress.progress(
                    (page_number + 1) / total
                )

            progress.empty()

            pdf_document.close()

        except Exception as e:

            st.error(
                f"Image extraction error: {e}"
            )


        st.success(
            f"🖼️ Found "
            f"{len(extracted_images)} images."
        )


        # ====================================================
        # STEP 7: BLIP IMAGE CAPTIONING
        # ====================================================

        if extracted_images:

            st.header(
                "👁️ 6. BLIP Image Understanding"
            )

            try:

                processor, blip_model = (
                    load_blip()
                )

                image_captions = []

                progress = st.progress(0)

                for number, item in enumerate(
                    extracted_images
                ):

                    try:

                        image = Image.open(
                            item["path"]
                        ).convert(
                            "RGB"
                        )

                        inputs = processor(
                            images=image,
                            return_tensors="pt"
                        )

                        output = blip_model.generate(
                            **inputs,
                            max_new_tokens=50
                        )

                        caption = (
                            processor.decode(
                                output[0],
                                skip_special_tokens=True
                            )
                        )

                    except Exception as e:

                        caption = (
                            "Image description unavailable"
                        )

                    item["caption"] = caption

                    image_captions.append(
                        caption
                    )

                    progress.progress(
                        (number + 1)
                        / len(extracted_images)
                    )

                progress.empty()

            except Exception as e:

                st.error(
                    f"BLIP error: {e}"
                )

                image_captions = []

            # =================================================
            # IMAGE FAISS
            # =================================================

            if image_captions:

                with st.spinner(
                    "Creating image embeddings..."
                ):

                    image_embeddings = (
                        embedding_model.encode(
                            image_captions,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        ).astype(
                            "float32"
                        )
                    )

                image_dimension = (
                    image_embeddings.shape[1]
                )

                image_faiss = faiss.IndexFlatIP(
                    image_dimension
                )

                image_faiss.add(
                    image_embeddings
                )

                faiss.write_index(
                    image_faiss,
                    IMAGE_INDEX_PATH
                )

                image_metadata = []

                for index_number, item in enumerate(
                    extracted_images
                ):

                    image_metadata.append(
                        {
                            "id": index_number,
                            "path": item.get(
                                "path",
                                ""
                            ),
                            "filename": item.get(
                                "filename",
                                os.path.basename(
                                    item.get(
                                        "path",
                                        ""
                                    )
                                )
                            ),
                            "page": item.get(
                                "page",
                                "Unknown"
                            ),
                            "caption": item.get(
                                "caption",
                                ""
                            )
                        }
                    )

                with open(
                    IMAGE_METADATA_PATH,
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        image_metadata,
                        f,
                        ensure_ascii=False,
                        indent=2
                    )

                st.success(
                    "✅ Image FAISS created."
                )

        else:

            st.info(
                "ℹ️ No embedded images found."
            )


        # ====================================================
        # COMPLETE
        # ====================================================

        st.session_state.processing_complete = True

        st.success(
            "🎉 PDF processing completed successfully!"
        )

        st.info(
            f"""
            ### 📊 Processing Summary

            **PDF:** {safe_filename}

            **Pages:** {total_pages}

            **Text chunks:** {len(chunks)}

            **Images:** {len(extracted_images)}

            **OCR:** {
                "Used"
                if st.session_state.ocr_used
                else "Not required"
            }
            """
        )

        st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

text_index, text_metadata = (
    load_text_database()
)

image_index, image_metadata = (
    load_image_database()
)

st.header(
    "📊 2. RAG Dashboard"
)

dashboard1, dashboard2, dashboard3, dashboard4 = (
    st.columns(4)
)

with dashboard1:

    st.metric(
        "PDF Pages",
        st.session_state.pdf_pages
    )

with dashboard2:

    st.metric(
        "Text Chunks",
        text_index.ntotal
        if text_index is not None
        else 0
    )

with dashboard3:

    st.metric(
        "Images",
        image_index.ntotal
        if image_index is not None
        else 0
    )

with dashboard4:

    st.metric(
        "Questions",
        len(st.session_state.history)
    )


# ============================================================
# ASK QUESTION
# ============================================================

st.header(
    "💬 3. Ask Questions"
)

if (
    text_index is None
    and image_index is None
):

    st.warning(
        "📄 Upload and process a PDF first."
    )

else:

    question = st.text_area(
        "Ask something about your PDF:",
        placeholder=(
            "Example: What is the main topic "
            "of this document?"
        ),
        height=100
    )

    col1, col2 = st.columns(2)

    with col1:

        text_top_k = st.slider(
            "Text results",
            min_value=1,
            max_value=10,
            value=5
        )

    with col2:

        image_top_k = st.slider(
            "Image results",
            min_value=1,
            max_value=5,
            value=3
        )


    # ========================================================
    # SEARCH
    # ========================================================

    if st.button(
        "🔎 Search + Generate Answer",
        type="primary",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            st.stop()


        # ====================================================
        # QUERY EMBEDDING
        # ====================================================

        query_embedding = (
            embedding_model.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(
                "float32"
            )
        )


        # ====================================================
        # TEXT RETRIEVAL
        # ====================================================

        text_results = []

        if (
            text_index is not None
            and text_index.ntotal > 0
        ):

            k = min(
                text_top_k,
                text_index.ntotal
            )

            scores, indices = (
                text_index.search(
                    query_embedding,
                    k
                )
            )

            for score, idx in zip(
                scores[0],
                indices[0]
            ):

                if (
                    idx >= 0
                    and idx < len(text_metadata)
                ):

                    item = text_metadata[idx]

                    text_results.append(
                        {
                            "text": item.get(
                                "text",
                                ""
                            ),
                            "page": item.get(
                                "page",
                                "Unknown"
                            ),
                            "source": item.get(
                                "source",
                                "PDF Text"
                            ),
                            "score": float(score)
                        }
                    )


        # ====================================================
        # IMAGE RETRIEVAL
        # ====================================================

        image_results = []

        if (
            image_index is not None
            and image_index.ntotal > 0
        ):

            k = min(
                image_top_k,
                image_index.ntotal
            )

            scores, indices = (
                image_index.search(
                    query_embedding,
                    k
                )
            )

            for score, idx in zip(
                scores[0],
                indices[0]
            ):

                if (
                    idx >= 0
                    and idx < len(image_metadata)
                ):

                    item = image_metadata[idx]

                    image_path = item.get(
                        "path",
                        ""
                    )

                    # Prevent KeyError: filename
                    filename = item.get(
                        "filename",
                        os.path.basename(
                            image_path
                        )
                    )

                    image_results.append(
                        {
                            "path": image_path,
                            "filename": filename,
                            "page": item.get(
                                "page",
                                "Unknown"
                            ),
                            "caption": item.get(
                                "caption",
                                ""
                            ),
                            "score": float(score)
                        }
                    )


        # ====================================================
        # RETRIEVED TEXT
        # ====================================================

        st.subheader(
            "📚 Retrieved Text / OCR"
        )

        if text_results:

            for number, result in enumerate(
                text_results
            ):

                with st.expander(
                    f"Text {number + 1} | "
                    f"Page {result['page']} | "
                    f"{result['source']} | "
                    f"Similarity: "
                    f"{result['score']:.4f}"
                ):

                    score = max(
                        0.0,
                        min(
                            1.0,
                            result["score"]
                        )
                    )

                    st.progress(
                        score
                    )

                    st.write(
                        result["text"]
                    )

        else:

            st.info(
                "No text retrieved."
            )


        # ====================================================
        # RETRIEVED IMAGES
        # ====================================================

        st.subheader(
            "🖼️ Retrieved Images"
        )

        if image_results:

            for number, result in enumerate(
                image_results
            ):

                with st.expander(
                    f"Image {number + 1} | "
                    f"Page {result['page']} | "
                    f"Similarity: "
                    f"{result['score']:.4f}"
                ):

                    image_path = result["path"]

                    if os.path.exists(
                        image_path
                    ):

                        st.image(
                            image_path,
                            width=500
                        )

                    else:

                        st.warning(
                            "Image file not found."
                        )

                    st.write(
                        f"**Filename:** "
                        f"{result['filename']}"
                    )

                    st.write(
                        f"**Page:** "
                        f"{result['page']}"
                    )

                    st.write(
                        f"**BLIP Description:** "
                        f"{result['caption']}"
                    )

                    score = max(
                        0.0,
                        min(
                            1.0,
                            result["score"]
                        )
                    )

                    st.progress(
                        score
                    )

        else:

            st.info(
                "No images retrieved."
            )


        # ====================================================
        # BUILD CONTEXT
        # ====================================================

        text_context = ""

        for result in text_results:

            text_context += (
                f"\n"
                f"[Page {result['page']} - "
                f"{result['source']}]\n"
                f"{result['text']}\n"
            )


        image_context = ""

        for result in image_results:

            image_context += (
                f"\n"
                f"[Image - Page "
                f"{result['page']}]\n"
                f"{result['caption']}\n"
            )


        # ====================================================
        # OLLAMA PROMPT
        # ====================================================

        prompt = f"""
You are an intelligent Multimodal RAG assistant.

Answer the user's question using ONLY the
retrieved information from the uploaded document.

========================
TEXT / OCR CONTEXT
========================

{text_context}

========================
IMAGE CONTEXT
========================

{image_context}

========================
USER QUESTION
========================

{question}

========================
RULES
========================

1. Do not invent information.
2. Use retrieved PDF text when relevant.
3. Use OCR information when relevant.
4. Use image descriptions when relevant.
5. Mention page numbers when possible.
6. Give a clear and direct answer.
7. If the information is unavailable, say:
   "I could not find this information in the uploaded document."
"""


        # ====================================================
        # OLLAMA
        # ====================================================

        st.subheader(
            "🤖 AI Generated Answer"
        )

        try:

            with st.spinner(
                "🧠 Ollama is generating the answer..."
            ):

                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=300
                )

            if response.status_code == 200:

                data = response.json()

                answer = data.get(
                    "response",
                    "No response received."
                )

                st.success(
                    "✅ Answer generated."
                )

                st.markdown(
                    "### 💡 Final Answer"
                )

                st.write(
                    answer
                )

            else:

                answer = (
                    f"Ollama returned HTTP "
                    f"{response.status_code}: "
                    f"{response.text}"
                )

                st.error(
                    answer
                )

        except requests.exceptions.ConnectionError:

            answer = (
                "❌ Cannot connect to Ollama.\n\n"
                "Make sure Ollama is running."
            )

            st.error(
                answer
            )

        except Exception as e:

            answer = (
                f"❌ Ollama error: {e}"
            )

            st.error(
                answer
            )


        # ====================================================
        # SOURCES
        # ====================================================

        st.subheader(
            "📌 Sources"
        )

        source_pages = []

        for result in text_results:

            page = str(
                result["page"]
            )

            if page not in source_pages:

                source_pages.append(
                    page
                )

        for result in image_results:

            page = str(
                result["page"]
            )

            if page not in source_pages:

                source_pages.append(
                    page
                )

        if source_pages:

            st.write(
                "Relevant pages: "
                + ", ".join(
                    source_pages
                )
            )

        else:

            st.write(
                "No source pages found."
            )


        # ====================================================
        # CHAT HISTORY
        # ====================================================

        st.session_state.history.append(
            {
                "question": question,
                "answer": answer
            }
        )


# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state.history:

    st.divider()

    st.header(
        "🕘 Question History"
    )

    for number, item in enumerate(
        reversed(
            st.session_state.history
        ),
        start=1
    ):

        with st.expander(
            f"Question {number}: "
            f"{item['question']}"
        ):

            st.write(
                item["answer"]
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Multimodal RAG System | "
    "700 MB PDF | "
    "FAISS | "
    "BLIP | "
    "Tesseract OCR | "
    "Ollama"
)