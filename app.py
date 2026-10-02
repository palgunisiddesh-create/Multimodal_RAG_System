import streamlit as st
import os
import json
import sqlite3
import hashlib
import time
from pathlib import Path

# ============================================================
# OPTIONAL / REQUIRED IMPORTS
# ============================================================

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

try:
    import pdfplumber
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

try:
    import fitz
    FITZ_AVAILABLE = True
except Exception:
    FITZ_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except Exception:
    EMBEDDING_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# BLIP
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    BLIP_AVAILABLE = True
except Exception:
    BLIP_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_DIR = BASE_DIR / "vectorstore"
IMAGE_DIR = BASE_DIR / "extracted_images"
OCR_DIR = BASE_DIR / "ocr_pages"

UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
OCR_DIR.mkdir(exist_ok=True)

TEXT_INDEX_PATH = VECTOR_DIR / "text_index.faiss"
TEXT_METADATA_PATH = VECTOR_DIR / "text_metadata.json"

IMAGE_INDEX_PATH = VECTOR_DIR / "image_index.faiss"
IMAGE_METADATA_PATH = VECTOR_DIR / "image_metadata.json"

DB_PATH = BASE_DIR / "users.db"


# ============================================================
# DATABASE
# ============================================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT,
            status TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS comparison_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            rag_answer TEXT,
            no_rag_answer TEXT,
            rag_time REAL,
            no_rag_time REAL,
            rag_chunks INTEGER,
            rag_images INTEGER,
            username TEXT,
            created_at TEXT
        )
    """)

    cursor.execute("SELECT * FROM users WHERE username='admin'")
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO users(username,password,role,status) VALUES(?,?,?,?)",
            (
                "admin",
                hash_password("admin123"),
                "admin",
                "active"
            )
        )

    cursor.execute("SELECT * FROM users WHERE username='user'")
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO users(username,password,role,status) VALUES(?,?,?,?)",
            (
                "user",
                hash_password("user123"),
                "user",
                "active"
            )
        )

    conn.commit()
    conn.close()


init_database()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""

if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = ""

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = ""

if "processed" not in st.session_state:
    st.session_state.processed = False

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "comparison_data" not in st.session_state:
    st.session_state.comparison_data = None

if "comparison_history" not in st.session_state:
    st.session_state.comparison_history = []


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #eef2ff 0%,
        #f5f3ff 40%,
        #fdf2f8 75%,
        #ecfeff 100%
    );
}

/* Main headings */

h1, h2, h3 {
    font-weight: 800 !important;
}

/* Buttons */

.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    border: none;
    padding: 10px 20px;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.18);
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #312e81,
        #4c1d95,
        #7e22ce
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Cards */

.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.10);
    margin-bottom: 20px;
}

.gradient-card {
    padding: 28px;
    border-radius: 22px;
    color: white;
    min-height: 170px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.16);
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.10);
}

.metric-number {
    font-size: 32px;
    font-weight: 800;
}

.metric-label {
    font-size: 15px;
    color: #666;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    # ========================================================
    # SIMPLE PROJECT POSTER - CREATED DIRECTLY IN app.py
    # ========================================================
    # No separate image file is required.
    # The poster is generated in memory and shown as a JPG-style image.

    if PIL_AVAILABLE:
        try:
            from PIL import ImageDraw, ImageFont

            width, height = 1400, 650
            poster = Image.new("RGB", (width, height), (239, 244, 255))
            draw = ImageDraw.Draw(poster)

            # Background gradient
            for y in range(height):
                r = int(239 - (y / height) * 35)
                g = int(244 - (y / height) * 45)
                b = int(255 - (y / height) * 5)
                draw.line([(0, y), (width, y)], fill=(r, g, b))

            # Decorative panels
            draw.rounded_rectangle(
                (45, 45, width - 45, height - 45),
                radius=35,
                fill=(255, 255, 255),
                outline=(99, 102, 241),
                width=4,
            )

            # Fonts
            try:
                title_font = ImageFont.truetype("arialbd.ttf", 58)
                subtitle_font = ImageFont.truetype("arialbd.ttf", 42)
                small_font = ImageFont.truetype("arial.ttf", 27)
                label_font = ImageFont.truetype("arialbd.ttf", 25)
            except Exception:
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
                label_font = ImageFont.load_default()

            # Main title
            draw.text(
                (90, 105),
                "MULTIMODAL RAG SYSTEM",
                font=title_font,
                fill=(31, 41, 95),
            )
            draw.text(
                (90, 180),
                "USING FAISS & OLLAMA",
                font=subtitle_font,
                fill=(79, 70, 229),
            )

            draw.text(
                (92, 255),
                "AI-powered document understanding and semantic retrieval",
                font=small_font,
                fill=(75, 85, 99),
            )

            # Simple document + AI visual
            doc_x, doc_y = 105, 345
            draw.rounded_rectangle(
                (doc_x, doc_y, doc_x + 230, doc_y + 190),
                radius=18,
                fill=(248, 250, 252),
                outline=(79, 70, 229),
                width=5,
            )
            draw.polygon(
                [
                    (doc_x + 165, doc_y),
                    (doc_x + 230, doc_y + 65),
                    (doc_x + 165, doc_y + 65),
                ],
                fill=(224, 231, 255),
            )
            for yy in (doc_y + 90, doc_y + 120, doc_y + 150):
                draw.rounded_rectangle(
                    (doc_x + 30, yy, doc_x + 190, yy + 8),
                    radius=4,
                    fill=(129, 140, 248),
                )

            # Flow arrows
            draw.line((350, 440, 495, 440), fill=(99, 102, 241), width=8)
            draw.polygon([(495, 440), (465, 420), (465, 460)], fill=(99, 102, 241))

            # AI circle
            cx, cy = 620, 440
            draw.ellipse((cx - 82, cy - 82, cx + 82, cy + 82), fill=(224, 231, 255), outline=(79, 70, 229), width=5)
            draw.ellipse((cx - 48, cy - 35, cx - 28, cy - 15), fill=(79, 70, 229))
            draw.ellipse((cx + 28, cy - 35, cx + 48, cy - 15), fill=(79, 70, 229))
            draw.arc((cx - 45, cy - 15, cx + 45, cy + 55), 20, 160, fill=(79, 70, 229), width=6)
            draw.text((cx - 33, cy + 78), "AI", font=label_font, fill=(31, 41, 95))

            # Flow to FAISS / Ollama
            draw.line((710, 440, 855, 440), fill=(99, 102, 241), width=8)
            draw.polygon([(855, 440), (825, 420), (825, 460)], fill=(99, 102, 241))

            draw.rounded_rectangle((880, 355, 1245, 525), radius=22, fill=(245, 243, 255), outline=(124, 58, 237), width=5)
            draw.text((930, 385), "FAISS", font=subtitle_font, fill=(109, 40, 217))
            draw.text((930, 445), "Vector Retrieval", font=small_font, fill=(75, 85, 99))

            # Bottom description
            draw.text(
                (90, 575),
                "PDF • OCR • Embeddings • FAISS • Ollama • Multimodal AI",
                font=small_font,
                fill=(55, 65, 81),
            )

            st.image(poster, use_container_width=True)

        except Exception:
            st.title("🤖 MULTIMODAL RAG SYSTEM USING FAISS & OLLAMA")
    else:
        st.title("🤖 MULTIMODAL RAG SYSTEM USING FAISS & OLLAMA")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ========================================================
    # CONNECT SYSTEM - EXISTING FUNCTIONALITY UNCHANGED
    # ========================================================
    left, center, right = st.columns([1, 2, 1])
    with center:
        if st.button(
            "🚀 Connect System",
            use_container_width=True,
            type="primary",
            key="home_connect_system",
        ):
            st.session_state.page = "login"
            st.rerun()

    st.markdown(
        "<div style='text-align:center; color:#6b7280; margin-top:16px;'>"
        "Artificial Intelligence and Data Science • Final Year Project • 2026"
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.title("🔐 Login")
    st.subheader("Multimodal RAG System using FAISS")
    st.write("Sign in as an administrator or user to access the system.")
    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if st.button("🔓 Login", use_container_width=True, type="primary"):
            if not username.strip() or not password:
                st.warning("Please enter both username and password.")
                return

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT username, role, status
                FROM users
                WHERE username=? AND password=?
                """,
                (username.strip(), hash_password(password))
            )
            result = cursor.fetchone()
            conn.close()

            if result:
                if result[2] != "active":
                    st.error("❌ Account is inactive.")
                    return

                st.session_state.logged_in = True
                st.session_state.username = result[0]
                st.session_state.role = result[1]
                st.session_state.page = "admin" if result[1] == "admin" else "user"
                st.success("✅ Login successful!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

        st.info("Demo Admin: admin / admin123\n\nDemo User: user / user123")

        if st.button("⬅️ Back to Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

def sidebar():

    with st.sidebar:
        st.title("🤖 Multimodal RAG")
        st.caption("Using FAISS and Large Language Models")
        st.divider()

        st.write(f"👤 **User:** {st.session_state.username}")
        st.write(f"🔑 **Role:** {st.session_state.role}")
        st.divider()

        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.page = "admin" if st.session_state.role == "admin" else "user"
            st.rerun()

        if st.button("📚 RAG Workspace", use_container_width=True):
            st.session_state.page = "rag"
            st.rerun()

        if st.button("📈 Evaluation & Results", use_container_width=True):
            st.session_state.page = "evaluation"
            st.rerun()

        if st.button("📊 Statistics", use_container_width=True):
            st.session_state.page = "statistics"
            st.rerun()

        if st.session_state.role == "admin":
            if st.button("👥 User Management", use_container_width=True):
                st.session_state.page = "users"
                st.rerun()

        st.divider()

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.role = ""
            st.session_state.page = "home"
            st.session_state.processed = False
            st.rerun()


# ============================================================
# COMPARISON HISTORY HELPERS
# ============================================================

def save_comparison_history(data):
    """Persist one measured RAG vs Without-RAG experiment."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO comparison_history
        (question, rag_answer, no_rag_answer, rag_time, no_rag_time,
         rag_chunks, rag_images, username, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data["query"],
            data["rag_answer"],
            data["no_rag_answer"],
            float(data["rag_time"]),
            float(data["no_rag_time"]),
            int(data["rag_chunks"]),
            int(data["rag_images"]),
            st.session_state.get("username", ""),
            data.get("created_at", time.strftime("%Y-%m-%d %H:%M:%S")),
        ),
    )
    conn.commit()
    conn.close()


def get_comparison_history():
    """Return comparison experiments, newest first."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, question, rag_answer, no_rag_answer, rag_time, no_rag_time,
               rag_chunks, rag_images, username, created_at
        FROM comparison_history
        ORDER BY id DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


# ============================================================
# ADMIN DASHBOARD
# ============================================================

def get_project_dashboard_status():
    """Collect live project status from files, FAISS indexes and session state."""
    pdf_files = [p for p in UPLOAD_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    image_files = [p for p in IMAGE_DIR.iterdir() if p.is_file()]

    chunk_count = 0
    if TEXT_METADATA_PATH.exists():
        try:
            with open(TEXT_METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            chunk_count = len(metadata) if isinstance(metadata, list) else 0
        except Exception:
            chunk_count = 0

    text_vectors = 0
    embedding_dimension = 0
    if FAISS_AVAILABLE and TEXT_INDEX_PATH.exists():
        try:
            text_index = faiss.read_index(str(TEXT_INDEX_PATH))
            text_vectors = int(text_index.ntotal)
            embedding_dimension = int(text_index.d)
        except Exception:
            text_vectors = 0

    image_vectors = 0
    if FAISS_AVAILABLE and IMAGE_INDEX_PATH.exists():
        try:
            image_index = faiss.read_index(str(IMAGE_INDEX_PATH))
            image_vectors = int(image_index.ntotal)
        except Exception:
            image_vectors = 0

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM comparison_history")
        experiment_count = int(cursor.fetchone()[0])
        conn.close()
    except Exception:
        experiment_count = 0

    return {
        "pdf_count": len(pdf_files),
        "current_pdf": st.session_state.uploaded_name if st.session_state.uploaded_name else (pdf_files[-1].name if pdf_files else "None"),
        "image_count": len(image_files),
        "chunk_count": chunk_count,
        "text_vectors": text_vectors,
        "image_vectors": image_vectors,
        "embedding_dimension": embedding_dimension,
        "embedding_ready": text_vectors > 0 or st.session_state.processed,
        "faiss_ready": TEXT_INDEX_PATH.exists() or IMAGE_INDEX_PATH.exists(),
        "text_faiss_ready": TEXT_INDEX_PATH.exists(),
        "image_faiss_ready": IMAGE_INDEX_PATH.exists(),
        "processed": bool(st.session_state.processed),
        "comparison": st.session_state.comparison_data,
        "experiment_count": experiment_count,
    }


def render_project_flow(status):
    st.subheader("🔄 Complete Project Workflow")

    flow_steps = [
        ("1", "Upload PDF", status["pdf_count"] > 0 or bool(st.session_state.uploaded_name)),
        ("2", "Extract Text and Images", status["processed"]),
        ("3", "OCR and Image Captioning", status["processed"]),
        ("4", f"Create Chunks ({status['chunk_count']})", status["chunk_count"] > 0),
        ("5", f"Generate Embeddings ({status['text_vectors']} vectors)", status["embedding_ready"]),
        ("6", "Store Vectors in FAISS", status["faiss_ready"]),
        ("7", "Ask a Question", bool(st.session_state.last_answer) or status["processed"]),
        ("8", "Generate RAG Answer", bool(st.session_state.last_answer)),
        ("9", "Generate Without-RAG Answer", bool(status["comparison"])),
        ("10", "Compare RAG vs Without-RAG", bool(status["comparison"])),
        ("11", "Display Performance Graph", bool(status["comparison"])),
    ]

    for i in range(0, len(flow_steps), 2):
        c1, c2 = st.columns(2)
        for col, step in zip((c1, c2), flow_steps[i:i + 2]):
            num, title, done = step
            with col:
                if done:
                    st.success(f"✅ {num}. {title}")
                else:
                    st.info(f"⏳ {num}. {title}")


def render_comparison_dashboard(status):
    st.subheader("⚖️ RAG vs Without-RAG Comparison")

    data = status["comparison"]
    if not data:
        st.info(
            "No comparison has been run yet. Open RAG Workspace, process a PDF, "
            "ask a question, and run the RAG vs Without-RAG comparison. "
            "The measured results and graph will then appear here."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("RAG Response Time", f"{data['rag_time']:.2f} s")
    with c2:
        st.metric("Without-RAG Response Time", f"{data['no_rag_time']:.2f} s")
    with c3:
        st.metric("RAG Text Retrieved", data["rag_chunks"])
    with c4:
        st.metric("RAG Images Retrieved", data["rag_images"])

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("### 🧠 RAG Answer")
        st.success(data["rag_answer"])
    with g2:
        st.markdown("### 💬 Without-RAG Answer")
        st.info(data["no_rag_answer"])

    st.markdown("### 📈 Response Time Graph")
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["RAG", "Without-RAG"],
                y=[data["rag_time"], data["no_rag_time"]],
                text=[
                    f"{data['rag_time']:.2f}s",
                    f"{data['no_rag_time']:.2f}s",
                ],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="RAG vs Without-RAG Response Time",
            xaxis_title="Method",
            yaxis_title="Response Time (seconds)",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(
            {
                "RAG": data["rag_time"],
                "Without-RAG": data["no_rag_time"],
            }
        )


def dashboard_overview(is_admin=False):
    sidebar()

    status = get_project_dashboard_status()

    title = "👨‍💼 Admin Dashboard" if is_admin else "👤 User Dashboard"
    st.title(title)
    st.caption("Multimodal RAG System using FAISS")
    st.write(
        f"Welcome, **{st.session_state.username}**. "
        "This dashboard shows the complete document-processing and RAG workflow."
    )

    # ========================================================
    # LIVE PROJECT STATUS
    # ========================================================
    st.subheader("📌 Project Status")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📄 PDF Uploaded", status["pdf_count"])
    with c2:
        st.metric("✂️ Text Chunks", status["chunk_count"])
    with c3:
        st.metric("🧠 Embeddings", status["text_vectors"])
    with c4:
        st.metric("⚡ FAISS Index", "Ready" if status["faiss_ready"] else "Not Ready")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("🖼️ Extracted Images", status["image_count"])
    with c6:
        st.metric("🔢 Embedding Dimension", status["embedding_dimension"] if status["embedding_dimension"] else "—")
    with c7:
        st.metric("🔍 OCR", "Available" if OCR_AVAILABLE else "Unavailable")
    with c8:
        st.metric("🤖 Ollama", "Available" if OLLAMA_AVAILABLE else "Unavailable")

    c9, c10 = st.columns(2)
    with c9:
        st.metric("🧪 Comparisons Run", status["experiment_count"])
    with c10:
        st.metric("📈 Evaluation", "Available" if status["experiment_count"] > 0 else "Waiting")

    # ========================================================
    # CURRENT DOCUMENT
    # ========================================================
    st.subheader("📄 Document Processing Status")

    d1, d2 = st.columns(2)
    with d1:
        st.write("**Current PDF:**", status["current_pdf"])
        st.write("**Processing Status:**", "✅ Completed" if status["processed"] else "⏳ Waiting for processing")
    with d2:
        st.write("**Text FAISS:**", "✅ Created" if status["text_faiss_ready"] else "⏳ Not created")
        st.write("**Image FAISS:**", "✅ Created" if status["image_faiss_ready"] else "⏳ Not created")

    # ========================================================
    # CORE COMPONENTS
    # ========================================================
    st.subheader("🧩 RAG System Components")
    components = [
        ("📄 PDF Processing", PDF_AVAILABLE, "Extract text from PDF documents"),
        ("🔍 OCR", OCR_AVAILABLE, "Read text from scanned pages"),
        ("🖼️ Image Understanding", BLIP_AVAILABLE, "Generate image captions"),
        ("✂️ Text Chunking", status["chunk_count"] > 0, "Split extracted text into chunks"),
        ("🧠 Embeddings", status["embedding_ready"], "Convert chunks into vector embeddings"),
        ("⚡ FAISS", status["faiss_ready"], "Store and retrieve vectors"),
        ("🤖 Ollama", OLLAMA_AVAILABLE, "Generate LLM answers"),
        ("📊 Comparison Graph", bool(status["comparison"]), "Visualize measured RAG vs Without-RAG results"),
        ("📈 Evaluation Dashboard", status["experiment_count"] > 0, "Review experiment metrics and history"),
    ]

    for i in range(0, len(components), 4):
        cols = st.columns(4)
        for col, (name, ready, desc) in zip(cols, components[i:i + 4]):
            with col:
                if ready:
                    st.success(f"✅ {name}")
                else:
                    st.warning(f"⏳ {name}")
                st.caption(desc)

    # ========================================================
    # ACTIONS
    # ========================================================
    st.subheader("🚀 Quick Actions")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        if st.button("📚 Open RAG Workspace", key=f"dashboard_rag_{'admin' if is_admin else 'user'}", use_container_width=True):
            st.session_state.page = "rag"
            st.rerun()
    with a2:
        if st.button("📈 Evaluation & Results", key=f"dashboard_eval_{'admin' if is_admin else 'user'}", use_container_width=True):
            st.session_state.page = "evaluation"
            st.rerun()
    with a3:
        if st.button("📊 Open Statistics", key=f"dashboard_stats_{'admin' if is_admin else 'user'}", use_container_width=True):
            st.session_state.page = "statistics"
            st.rerun()
    with a4:
        if is_admin:
            if st.button("👥 User Management", key="dashboard_users_admin", use_container_width=True):
                st.session_state.page = "users"
                st.rerun()
        else:
            st.info("Use RAG Workspace to process documents.")

    # ========================================================
    # COMPLETE FLOW
    # ========================================================
    render_project_flow(status)

    # ========================================================
    # COMPARISON + GRAPH
    # ========================================================
    render_comparison_dashboard(status)


def admin_dashboard():
    dashboard_overview(is_admin=True)


def user_dashboard():
    dashboard_overview(is_admin=False)


def extract_text_from_pdf(pdf_path):

    text = ""

    if PDF_AVAILABLE:

        try:

            with pdfplumber.open(pdf_path) as pdf:

                for page in pdf.pages:

                    page_text = page.extract_text()

                    if page_text:
                        text += page_text + "\n"

        except Exception:
            pass

    if not text.strip() and FITZ_AVAILABLE:

        try:

            document = fitz.open(pdf_path)

            for page in document:

                text += page.get_text() + "\n"

            document.close()

        except Exception:
            pass

    return text.strip()


# ============================================================
# IMAGE EXTRACTION
# ============================================================

def extract_images_from_pdf(pdf_path):

    extracted = []

    if not FITZ_AVAILABLE:
        return extracted

    try:

        document = fitz.open(pdf_path)

        for page_number in range(len(document)):

            page = document[page_number]

            images = page.get_images(full=True)

            for image_number, img in enumerate(images):

                try:

                    xref = img[0]

                    base_image = document.extract_image(xref)

                    image_bytes = base_image["image"]
                    extension = base_image["ext"]

                    filename = (
                        f"page_{page_number + 1}_"
                        f"image_{image_number + 1}.{extension}"
                    )

                    image_path = IMAGE_DIR / filename

                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    extracted.append(str(image_path))

                except Exception:
                    continue

        document.close()

    except Exception:
        pass

    return extracted


# ============================================================
# OCR
# ============================================================

def perform_ocr(pdf_path):

    ocr_text = ""

    if not FITZ_AVAILABLE or not OCR_AVAILABLE:
        return ocr_text

    try:

        document = fitz.open(pdf_path)

        for page_number in range(len(document)):

            page = document[page_number]

            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))

            image_path = OCR_DIR / (
                f"ocr_page_{page_number + 1}.png"
            )

            pix.save(str(image_path))

            if PIL_AVAILABLE:

                image = Image.open(image_path)

                text = pytesseract.image_to_string(image)

                if text.strip():

                    ocr_text += (
                        f"\n[OCR Page {page_number + 1}]\n"
                        f"{text}\n"
                    )

        document.close()

    except Exception as e:

        st.warning(
            f"OCR warning: {str(e)}"
        )

    return ocr_text.strip()


# ============================================================
# BLIP
# ============================================================

@st.cache_resource
def load_blip():

    if not BLIP_AVAILABLE:
        return None, None

    try:

        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        return processor, model

    except Exception:

        return None, None


def generate_image_caption(image_path):

    if not BLIP_AVAILABLE:
        return "Image extracted from document."

    try:

        processor, model = load_blip()

        if processor is None or model is None:
            return "Image extracted from document."

        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=40
            )

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption

    except Exception:

        return "Image extracted from document."


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=500,
    overlap=80
):

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

        if end >= len(text):
            break

    return chunks


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    if not EMBEDDING_AVAILABLE:
        return None

    try:

        return SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    except Exception:

        return None


# ============================================================
# TEXT FAISS
# ============================================================

def create_text_faiss(chunks):

    if not FAISS_AVAILABLE or not EMBEDDING_AVAILABLE:
        return False

    model = load_embedding_model()

    if model is None or not chunks:
        return False

    try:

        embeddings = model.encode(
            chunks,
            convert_to_numpy=True
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings.astype("float32"))

        faiss.write_index(
            index,
            str(TEXT_INDEX_PATH)
        )

        metadata = []

        for i, chunk in enumerate(chunks):

            metadata.append({
                "id": i,
                "text": chunk
            })

        with open(
            TEXT_METADATA_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False
            )

        return True

    except Exception as e:

        st.error(
            f"Text FAISS error: {e}"
        )

        return False


# ============================================================
# IMAGE FAISS
# ============================================================

def create_image_faiss(image_paths):

    if not FAISS_AVAILABLE or not EMBEDDING_AVAILABLE:
        return False

    if not image_paths:
        return False

    model = load_embedding_model()

    if model is None:
        return False

    try:

        captions = []

        for path in image_paths:

            caption = generate_image_caption(path)

            captions.append(caption)

        embeddings = model.encode(
            captions,
            convert_to_numpy=True
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(
            embeddings.astype("float32")
        )

        faiss.write_index(
            index,
            str(IMAGE_INDEX_PATH)
        )

        metadata = []

        for path, caption in zip(
            image_paths,
            captions
        ):

            metadata.append({
                "image_path": path,
                "caption": caption
            })

        with open(
            IMAGE_METADATA_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False
            )

        return True

    except Exception as e:

        st.warning(
            f"Image FAISS warning: {e}"
        )

        return False


# ============================================================
# RETRIEVE TEXT
# ============================================================

def retrieve_text(
    query,
    top_k=3
):

    if not FAISS_AVAILABLE:
        return []

    if not TEXT_INDEX_PATH.exists():
        return []

    if not TEXT_METADATA_PATH.exists():
        return []

    model = load_embedding_model()

    if model is None:
        return []

    try:

        index = faiss.read_index(
            str(TEXT_INDEX_PATH)
        )

        with open(
            TEXT_METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True
        )

        distances, indices = index.search(
            query_embedding.astype("float32"),
            min(top_k, index.ntotal)
        )

        results = []

        for distance, idx in zip(
            distances[0],
            indices[0]
        ):

            if idx >= 0 and idx < len(metadata):

                results.append({
                    "text": metadata[idx]["text"],
                    "distance": float(distance)
                })

        return results

    except Exception:

        return []


# ============================================================
# RETRIEVE IMAGES
# ============================================================

def retrieve_images(
    query,
    top_k=3
):

    if not FAISS_AVAILABLE:
        return []

    if not IMAGE_INDEX_PATH.exists():
        return []

    if not IMAGE_METADATA_PATH.exists():
        return []

    model = load_embedding_model()

    if model is None:
        return []

    try:

        index = faiss.read_index(
            str(IMAGE_INDEX_PATH)
        )

        with open(
            IMAGE_METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True
        )

        distances, indices = index.search(
            query_embedding.astype("float32"),
            min(top_k, index.ntotal)
        )

        results = []

        for distance, idx in zip(
            distances[0],
            indices[0]
        ):

            if idx >= 0 and idx < len(metadata):

                results.append({
                    "image_path": metadata[idx]["image_path"],
                    "caption": metadata[idx]["caption"],
                    "distance": float(distance)
                })

        return results

    except Exception:

        return []


# ============================================================
# OLLAMA RAG
# ============================================================

def generate_answer(
    query,
    context
):

    if not OLLAMA_AVAILABLE:

        return (
            "Ollama Python package is not installed. "
            "Please install it using: pip install ollama"
        )

    if not context.strip():

        return (
            "No relevant information was retrieved "
            "from the uploaded document."
        )

    prompt = f"""

You are a helpful Multimodal RAG assistant.

Answer the user's question using ONLY the
provided retrieved context.

If the answer is not present in the context,
clearly say that the information is not
available in the retrieved document.

Retrieved Context:
{context}

Question:
{query}

Give a clear, simple and useful answer.

"""

    try:

        response = ollama.chat(
            model="llama3.2:latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        return f"Ollama error: {e}"


# ============================================================
# WITHOUT RAG
# ============================================================

def generate_without_rag(
    query
):

    if not OLLAMA_AVAILABLE:

        return (
            "Ollama Python package is not installed."
        )

    prompt = f"""

Answer the following question directly using
your general language-model knowledge.

Do NOT retrieve information from the uploaded
document.

Question:
{query}

Give a clear and simple answer.

"""

    try:

        response = ollama.chat(
            model="llama3.2:latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        return f"Ollama error: {e}"


# ============================================================
# RAG WORKSPACE
# ============================================================

def rag_workspace():

    sidebar()

    st.title("🧠 Multimodal RAG Workspace")
    st.caption(
        "Upload documents, process multimodal information, ask questions, "
        "and compare RAG with Without-RAG."
    )

    # ========================================================
    # STEP 1 — UPLOAD AND PROCESS
    # ========================================================

    st.markdown("## 📤 Step 1 — Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        key="rag_pdf_uploader"
    )

    if uploaded_file:

        file_path = UPLOAD_DIR / uploaded_file.name

        # Avoid rewriting the same file on every Streamlit rerun.
        current_path = st.session_state.get("uploaded_path", "")
        if current_path != str(file_path) or not file_path.exists():
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.uploaded_path = str(file_path)
            st.session_state.uploaded_name = uploaded_file.name
            st.session_state.processed = False
            st.session_state.last_answer = ""
            st.session_state.rag_query = ""
            st.session_state.rag_result = None
            st.session_state.comparison_data = None

        st.success(f"✅ PDF uploaded: {uploaded_file.name}")

        if st.button(
            "⚙️ Process Document",
            use_container_width=True,
            type="primary"
        ):

            progress = st.progress(0)
            status_text = st.empty()

            try:
                # TEXT
                status_text.write("📄 Extracting text from PDF...")
                text = extract_text_from_pdf(file_path)
                progress.progress(20)

                # OCR
                status_text.write("🔍 Performing OCR...")
                ocr_text = perform_ocr(file_path)
                progress.progress(40)

                combined_text = text.strip()
                if ocr_text.strip():
                    combined_text += "\n\n" + ocr_text.strip()

                # CHUNKS
                status_text.write("✂️ Creating text chunks...")
                chunks = create_chunks(
                    combined_text,
                    chunk_size=500,
                    overlap=80
                )
                progress.progress(55)

                # IMAGES
                status_text.write("🖼️ Extracting images...")
                image_paths = extract_images_from_pdf(file_path)
                progress.progress(70)

                # TEXT FAISS
                status_text.write("🧠 Generating embeddings and creating Text FAISS...")
                text_success = create_text_faiss(chunks)
                progress.progress(82)

                # IMAGE FAISS
                status_text.write("🤖 Creating Image FAISS and captions...")
                image_success = create_image_faiss(image_paths)
                progress.progress(100)

                st.session_state.processed = True
                st.session_state.last_answer = ""
                st.session_state.rag_query = ""
                st.session_state.rag_result = None
                st.session_state.comparison_data = None

                status_text.empty()
                st.success("🎉 Document processing completed!")

                # RESULTS
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric("Text Characters", len(combined_text))

                with c2:
                    st.metric("Text Chunks", len(chunks))

                with c3:
                    st.metric("Extracted Images", len(image_paths))

                with c4:
                    st.metric(
                        "OCR",
                        "Available" if OCR_AVAILABLE else "Unavailable"
                    )

                c5, c6, c7, c8 = st.columns(4)

                with c5:
                    st.metric(
                        "Embeddings",
                        len(chunks)
                    )

                with c6:
                    st.metric(
                        "Text FAISS",
                        "Ready" if text_success else "Failed"
                    )

                with c7:
                    st.metric(
                        "Image FAISS",
                        "Ready" if image_success else "Failed"
                    )

                with c8:
                    if FAISS_AVAILABLE and TEXT_INDEX_PATH.exists():
                        try:
                            idx = faiss.read_index(str(TEXT_INDEX_PATH))
                            st.metric("Embedding Dimension", int(idx.d))
                        except Exception:
                            st.metric("Embedding Dimension", "—")
                    else:
                        st.metric("Embedding Dimension", "—")

            except Exception as e:
                progress.empty()
                status_text.empty()
                st.error(f"Document processing failed: {e}")

    # ========================================================
    # STEP 2 — ASK QUESTION
    # ========================================================

    if st.session_state.processed:

        st.markdown("---")
        st.markdown("## 💬 Step 2 — Ask a Question")

        query = st.text_input(
            "Enter your question",
            value=st.session_state.get("rag_query", ""),
            key="rag_query_input",
            placeholder="Example: What is the main topic of this document?"
        )

        # Store the current query so it is available for comparison
        # even after Streamlit reruns.
        if query.strip():
            st.session_state.rag_query = query.strip()

        if st.button(
            "🤖 Ask RAG",
            use_container_width=True,
            type="primary"
        ):

            if not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving information and generating RAG answer..."):
                    start_time = time.time()

                    text_results = retrieve_text(query, 3)
                    image_results = retrieve_images(query, 3)

                    retrieval_time = time.time() - start_time

                    context_parts = []

                    for result in text_results:
                        context_parts.append(result["text"])

                    for result in image_results:
                        context_parts.append(
                            "Image description: " + result["caption"]
                        )

                    context = "\n\n".join(context_parts)

                    answer = generate_answer(query, context)
                    total_time = time.time() - start_time

                    st.session_state.rag_query = query.strip()
                    st.session_state.last_answer = answer
                    st.session_state.rag_result = {
                        "query": query.strip(),
                        "answer": answer,
                        "retrieval_time": retrieval_time,
                        "total_time": total_time,
                        "text_results": text_results,
                        "image_results": image_results,
                    }

        # ----------------------------------------------------
        # SHOW RAG ANSWER FROM SAVED SESSION DATA
        # ----------------------------------------------------

        rag_result = st.session_state.get("rag_result")

        if rag_result:

            st.subheader("🤖 RAG Answer")
            st.success(rag_result["answer"])

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Retrieved Text",
                    len(rag_result["text_results"])
                )

            with c2:
                st.metric(
                    "Retrieved Images",
                    len(rag_result["image_results"])
                )

            with c3:
                st.metric(
                    "Response Time",
                    f"{rag_result['total_time']:.2f}s"
                )

            # ------------------------------------------------
            # RAG RETRIEVED SOURCES
            # ------------------------------------------------

            st.markdown("### 📚 Retrieved Text Sources")

            if rag_result["text_results"]:
                for i, result in enumerate(
                    rag_result["text_results"],
                    start=1
                ):
                    with st.expander(
                        f"Text Source {i} | Distance: {result['distance']:.4f}"
                    ):
                        st.write(result["text"])
            else:
                st.info("No text sources were retrieved.")

            if rag_result["image_results"]:
                st.markdown("### 🖼️ Retrieved Images")
                cols = st.columns(len(rag_result["image_results"]))

                for col, result in zip(
                    cols,
                    rag_result["image_results"]
                ):
                    with col:
                        image_path = result.get("image_path", "")
                        if image_path and os.path.exists(image_path):
                            st.image(
                                image_path,
                                use_container_width=True
                            )
                        st.caption(result.get("caption", "No caption"))

        # ====================================================
        # STEP 3 — RAG VS WITHOUT-RAG
        # ====================================================

        if st.session_state.get("rag_query", "").strip():

            st.markdown("---")
            st.markdown("## ⚖️ Step 3 — RAG vs Without-RAG")
            st.info(
                "The same question will be used for both approaches. "
                "RAG retrieves document context through FAISS; Without-RAG "
                "sends the question directly to Ollama."
            )

            st.write(
                f"**Comparison question:** {st.session_state.rag_query}"
            )

            if st.button(
                "📊 Run RAG vs Without-RAG Comparison",
                key="run_comparison_button",
                use_container_width=True,
                type="secondary"
            ):

                compare_query = st.session_state.rag_query.strip()

                with st.spinner(
                    "Running both approaches and measuring actual response times..."
                ):

                    # -------------------------------
                    # RAG
                    # -------------------------------
                    rag_start = time.perf_counter()

                    rag_text_results = retrieve_text(
                        compare_query,
                        3
                    )

                    rag_image_results = retrieve_images(
                        compare_query,
                        3
                    )

                    rag_context_parts = []

                    for result in rag_text_results:
                        rag_context_parts.append(result["text"])

                    for result in rag_image_results:
                        rag_context_parts.append(
                            "Image description: " + result["caption"]
                        )

                    rag_context = "\n\n".join(rag_context_parts)

                    rag_answer = generate_answer(
                        compare_query,
                        rag_context
                    )

                    rag_time = time.perf_counter() - rag_start

                    # -------------------------------
                    # WITHOUT RAG
                    # -------------------------------
                    no_rag_start = time.perf_counter()

                    no_rag_answer = generate_without_rag(
                        compare_query
                    )

                    no_rag_time = time.perf_counter() - no_rag_start

                    # Save measured experiment in session state
                    st.session_state.comparison_data = {
                        "query": compare_query,
                        "rag_time": rag_time,
                        "no_rag_time": no_rag_time,
                        "rag_chunks": len(rag_text_results),
                        "rag_images": len(rag_image_results),
                        "rag_answer": rag_answer,
                        "no_rag_answer": no_rag_answer,
                        "created_at": time.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    }

                    try:
                        save_comparison_history(st.session_state.comparison_data)
                    except Exception as save_error:
                        st.warning(
                            f"Comparison completed, but history could not be saved: {save_error}"
                        )

                st.success(
                    "✅ Comparison completed successfully. "
                    "Measured results are now available on this page and the dashboard."
                )

        # ====================================================
        # SHOW COMPARISON RESULTS
        # ====================================================

        if st.session_state.get("comparison_data"):

            data = st.session_state.comparison_data

            st.markdown("---")
            st.subheader("📊 RAG vs Without-RAG Analysis")
            st.write(f"**Question:** {data['query']}")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "RAG Response Time",
                    f"{data['rag_time']:.2f}s"
                )

            with c2:
                st.metric(
                    "Without-RAG Response Time",
                    f"{data['no_rag_time']:.2f}s"
                )

            with c3:
                st.metric(
                    "RAG Text Sources",
                    data["rag_chunks"]
                )

            with c4:
                st.metric(
                    "RAG Images",
                    data["rag_images"]
                )

            # Answers
            a1, a2 = st.columns(2)

            with a1:
                st.markdown("### 🧠 With RAG")
                st.success(data["rag_answer"])

            with a2:
                st.markdown("### 💬 Without-RAG")
                st.info(data["no_rag_answer"])

            # Graph
            st.markdown("### 📈 Response Time Comparison")

            if PLOTLY_AVAILABLE:

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=["RAG", "Without-RAG"],
                        y=[
                            data["rag_time"],
                            data["no_rag_time"]
                        ],
                        text=[
                            f"{data['rag_time']:.2f}s",
                            f"{data['no_rag_time']:.2f}s"
                        ],
                        textposition="auto"
                    )
                )

                fig.update_layout(
                    title="RAG vs Without-RAG Response Time",
                    xaxis_title="Method",
                    yaxis_title="Response Time (seconds)",
                    height=450
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.bar_chart(
                    {
                        "RAG": data["rag_time"],
                        "Without-RAG": data["no_rag_time"]
                    }
                )

            st.info(
                "ℹ️ The graph uses actual measured response times from the "
                "current experiment. No accuracy values are invented."
            )


# ============================================================
# EVALUATION & RESULTS DASHBOARD
# ============================================================

def evaluation_page():

    sidebar()

    st.title("📈 Evaluation & Results")
    st.caption(
        "Measured evaluation of RAG and Without-RAG using the experiments "
        "run in the system."
    )

    rows = get_comparison_history()

    current = st.session_state.get("comparison_data")

    if current and (not rows or rows[0][1] != current.get("query")):
        # Keep the UI immediately useful even if the DB write was interrupted.
        st.info("A comparison is available in the current session.")

    st.markdown("## 📌 Evaluation Overview")

    if not rows and not current:
        st.info(
            "No evaluation experiments are available yet. Open RAG Workspace, "
            "process a PDF, ask a question, and run the RAG vs Without-RAG comparison."
        )
        st.markdown("### Recommended test question")
        st.code(
            "What is the main topic of this document and what are its key objectives?"
        )
        return

    # Build display data. Current session result is always represented.
    experiments = []
    for row in rows:
        (
            comparison_id, question, rag_answer, no_rag_answer, rag_time,
            no_rag_time, rag_chunks, rag_images, username, created_at
        ) = row
        experiments.append({
            "id": comparison_id,
            "question": question,
            "rag_answer": rag_answer,
            "no_rag_answer": no_rag_answer,
            "rag_time": float(rag_time),
            "no_rag_time": float(no_rag_time),
            "rag_chunks": int(rag_chunks),
            "rag_images": int(rag_images),
            "username": username,
            "created_at": created_at,
        })

    if current:
        current_experiment = {
            "id": "Current",
            "question": current["query"],
            "rag_answer": current["rag_answer"],
            "no_rag_answer": current["no_rag_answer"],
            "rag_time": float(current["rag_time"]),
            "no_rag_time": float(current["no_rag_time"]),
            "rag_chunks": int(current["rag_chunks"]),
            "rag_images": int(current["rag_images"]),
            "username": st.session_state.get("username", ""),
            "created_at": current.get("created_at", ""),
        }
        if not experiments or experiments[0]["question"] != current_experiment["question"]:
            experiments.insert(0, current_experiment)

    latest = experiments[0]

    # ---------------------------------------------------------
    # KEY METRICS
    # ---------------------------------------------------------
    st.markdown("## 🔬 Latest Experiment")
    st.write(f"**Question:** {latest['question']}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("RAG Response Time", f"{latest['rag_time']:.2f} s")
    with c2:
        st.metric("Without-RAG Time", f"{latest['no_rag_time']:.2f} s")
    with c3:
        st.metric("Text Sources", latest["rag_chunks"])
    with c4:
        st.metric("Images Retrieved", latest["rag_images"])
    with c5:
        st.metric("Experiments", len(rows))

    # ---------------------------------------------------------
    # ANSWERS
    # ---------------------------------------------------------
    st.markdown("## 🤖 Answer Comparison")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("### 🧠 With RAG")
        st.success(latest["rag_answer"])
    with a2:
        st.markdown("### 💬 Without-RAG")
        st.info(latest["no_rag_answer"])

    # ---------------------------------------------------------
    # GRAPHS
    # ---------------------------------------------------------
    st.markdown("## 📊 Evaluation Graphs")

    if PLOTLY_AVAILABLE:
        # Graph 1: latest response time
        fig1 = go.Figure()
        fig1.add_trace(
            go.Bar(
                x=["With RAG", "Without-RAG"],
                y=[latest["rag_time"], latest["no_rag_time"]],
                text=[
                    f"{latest['rag_time']:.2f}s",
                    f"{latest['no_rag_time']:.2f}s",
                ],
                textposition="auto",
                name="Response Time",
            )
        )
        fig1.update_layout(
            title="Latest Response Time Comparison",
            xaxis_title="Method",
            yaxis_title="Response Time (seconds)",
            height=420,
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Graph 2: retrieval sources from latest experiment
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=["Text Sources", "Images"],
                y=[latest["rag_chunks"], latest["rag_images"]],
                text=[latest["rag_chunks"], latest["rag_images"]],
                textposition="auto",
                name="Retrieved Sources",
            )
        )
        fig2.update_layout(
            title="Latest RAG Retrieval Sources",
            xaxis_title="Source Type",
            yaxis_title="Count",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Graph 3: experiment history response times
        history_for_graph = list(reversed(experiments))
        if len(history_for_graph) >= 2:
            x_labels = [f"Experiment {i + 1}" for i in range(len(history_for_graph))]
            fig3 = go.Figure()
            fig3.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=[e["rag_time"] for e in history_for_graph],
                    mode="lines+markers+text",
                    text=[f"{e['rag_time']:.2f}s" for e in history_for_graph],
                    textposition="top center",
                    name="RAG",
                )
            )
            fig3.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=[e["no_rag_time"] for e in history_for_graph],
                    mode="lines+markers+text",
                    text=[f"{e['no_rag_time']:.2f}s" for e in history_for_graph],
                    textposition="bottom center",
                    name="Without-RAG",
                )
            )
            fig3.update_layout(
                title="Response Time Across Experiments",
                xaxis_title="Experiment",
                yaxis_title="Response Time (seconds)",
                height=420,
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Run at least two experiments to display the experiment-history graph.")

    else:
        st.warning("Plotly is not available. The detailed interactive graphs are disabled.")

    # ---------------------------------------------------------
    # SUMMARY TABLE
    # ---------------------------------------------------------
    st.markdown("## 📋 Comparison History")

    table_data = []
    for e in experiments:
        table_data.append({
            "ID": e["id"],
            "Question": e["question"],
            "RAG Time (s)": round(e["rag_time"], 2),
            "Without-RAG Time (s)": round(e["no_rag_time"], 2),
            "Text Sources": e["rag_chunks"],
            "Images": e["rag_images"],
            "User": e["username"],
            "Date": e["created_at"],
        })

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
    )

    # ---------------------------------------------------------
    # INTERPRETATION NOTES
    # ---------------------------------------------------------
    st.markdown("## 📝 Evaluation Notes")
    st.info(
        "The graphs use actual measured response times and retrieved-source "
        "counts from your experiments. The application does not invent accuracy "
        "scores. Response time can vary between runs because local model loading, "
        "retrieval and generation costs vary."
    )

    st.success(
        "✅ Evaluation dashboard completed. Run more questions to build a larger "
        "experimental history for your final-year project."
    )


# ============================================================
# STATISTICS
# ============================================================

def statistics_page():

    sidebar()

    st.title("📊 System Statistics")
    st.caption("Current system and project statistics.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM users"
    )

    total_users = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM users WHERE role='admin'"
    )

    admins = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM users WHERE role='user'"
    )

    users = cursor.fetchone()[0]

    conn.close()

    pdfs = list(
        UPLOAD_DIR.glob("*")
    )

    images = list(
        IMAGE_DIR.glob("*")
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "👥 Users",
            total_users
        )

    with c2:

        st.metric(
            "👨‍💼 Admins",
            admins
        )

    with c3:

        st.metric(
            "📄 Files",
            len(pdfs)
        )

    with c4:

        st.metric(
            "🖼️ Images",
            len(images)
        )

    st.markdown("---")

    st.subheader(
        "⚙️ Component Status"
    )

    status_data = {

        "FAISS": FAISS_AVAILABLE,

        "PDF Processing": PDF_AVAILABLE,

        "PyMuPDF": FITZ_AVAILABLE,

        "OCR": OCR_AVAILABLE,

        "Embeddings": EMBEDDING_AVAILABLE,

        "Ollama": OLLAMA_AVAILABLE,

        "Plotly": PLOTLY_AVAILABLE,

        "BLIP": BLIP_AVAILABLE

    }

    for component, available in status_data.items():

        if available:

            st.success(
                f"✅ {component}: Available"
            )

        else:

            st.warning(
                f"⚠️ {component}: Not Available"
            )


# ============================================================
# USER MANAGEMENT
# ============================================================

def user_management():

    sidebar()

    if st.session_state.role != "admin":

        st.error(
            "Access denied."
        )

        return

    st.title("👥 User Management")
    st.caption("Create and manage system users.")

    st.subheader(
        "➕ Add New User"
    )

    c1, c2 = st.columns(2)

    with c1:

        new_username = st.text_input(
            "Username"
        )

    with c2:

        new_password = st.text_input(
            "Password",
            type="password"
        )

    role = st.selectbox(
        "Role",
        [
            "user",
            "admin"
        ]
    )

    if st.button(
        "➕ CREATE USER",
        use_container_width=True
    ):

        if not new_username or not new_password:

            st.warning(
                "Please enter username and password."
            )

        else:

            try:

                conn = sqlite3.connect(
                    DB_PATH
                )

                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO users
                    (username,password,role,status)
                    VALUES(?,?,?,?)
                    """,
                    (
                        new_username,
                        hash_password(new_password),
                        role,
                        "active"
                    )
                )

                conn.commit()
                conn.close()

                st.success(
                    "✅ User created successfully!"
                )

            except sqlite3.IntegrityError:

                st.error(
                    "❌ Username already exists."
                )

    st.markdown("---")

    st.subheader(
        "📋 Existing Users"
    )

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, username, role, status
        FROM users
        ORDER BY id
        """
    )

    rows = cursor.fetchall()

    conn.close()

    if rows:

        for row in rows:

            uid, username, role, status = row

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.write(
                    f"**ID:** {uid}"
                )

            with c2:
                st.write(
                    f"👤 {username}"
                )

            with c3:
                st.write(
                    f"🔑 {role}"
                )

            with c4:
                st.write(
                    f"🟢 {status}"
                )

            st.markdown("---")


# ============================================================
# MAIN ROUTER
# ============================================================

# IMPORTANT:
# This guarantees that the colourful home dashboard
# is shown before login.

if not st.session_state.logged_in:

    if st.session_state.page == "home":

        home_page()

    elif st.session_state.page == "login":

        login_page()

    else:

        st.session_state.page = "home"
        st.rerun()

else:

    if (
        st.session_state.role != "admin"
        and st.session_state.page == "users"
    ):

        st.session_state.page = "user"
        st.rerun()

    if st.session_state.page == "admin":

        admin_dashboard()

    elif st.session_state.page == "user":

        user_dashboard()

    elif st.session_state.page == "rag":

        rag_workspace()

    elif st.session_state.page == "evaluation":

        evaluation_page()

    elif st.session_state.page == "statistics":

        statistics_page()

    elif st.session_state.page == "users":

        user_management()

    else:

        if st.session_state.role == "admin":

            st.session_state.page = "admin"

        else:

            st.session_state.page = "user"

        st.rerun()