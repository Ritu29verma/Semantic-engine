import os
# 🚫 Make sure HF never auto-uses accelerate/device_map ("meta" tensors)
os.environ.setdefault("TRANSFORMERS_USE_ACCELERATE", "0")

import io
import pickle
import faiss
import re
import json
import requests
import torch
import numpy as np
import streamlit as st
from tqdm import tqdm
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
)

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast & CPU friendly (384 dims)
INDEX_DIR = "data/index"
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")

os.makedirs(INDEX_DIR, exist_ok=True)

# ---------------------------
# Utilities
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str = MODEL_NAME):
    return SentenceTransformer(model_name)

def extract_text_from_pdf(file_like) -> list[dict]:
    """Return list of {'page': int, 'text': str} for each page."""
    reader = PdfReader(file_like)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages

def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for p in pages:
        for ch in splitter.split_text(p["text"]):
            chunks.append({"page": p["page"], "text": ch})
    return chunks

def normalize(vecs: np.ndarray) -> np.ndarray:
    """L2 normalize for cosine similarity using inner product."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

def save_index(index, metadata):
    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

def load_index():
    if not (os.path.exists(FAISS_PATH) and os.path.exists(META_PATH)):
        return None, []
    index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def new_index(dimension: int):
    return faiss.IndexFlatIP(dimension)

# ---------------------------
# Ingestion & Indexing
# ---------------------------
def build_or_update_index(uploaded_files: list):
    model = load_embedder()
    dim = model.get_sentence_embedding_dimension()

    # Load or create index
    index, metadata = load_index()
    if index is None:
        index = new_index(dim)
        metadata = []

    # Read, chunk, embed
    new_vecs = []
    new_meta = []

    for up in uploaded_files:
        # Read file bytes for PyPDF
        file_bytes = up.read()
        pages = extract_text_from_pdf(io.BytesIO(file_bytes))
        chunks = chunk_pages(pages)
        texts = [c["text"] for c in chunks]
        if not texts:
            continue

        # Embed & normalize
        embs = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        embs = embs.astype("float32")
        embs = normalize(embs)

        # Add to staging
        for c, e in zip(chunks, embs):
            new_vecs.append(e)
            new_meta.append({
                "filename": up.name,
                "page": c["page"],
                "text": c["text"]
            })

    if new_vecs:
        new_vecs_np = np.vstack(new_vecs)
        index.add(new_vecs_np)
        metadata.extend(new_meta)
        save_index(index, metadata)
        return len(new_meta)
    return 0

# ---------------------------
# Search
# ---------------------------
def search(query: str, k: int = 5):
    index, metadata = load_index()
    if index is None or index.ntotal == 0:
        return []

    model = load_embedder()
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    q = normalize(q)
    scores, ids = index.search(q, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        m = metadata[idx]
        results.append({
            "score": float(score),
            "filename": m["filename"],
            "page": m["page"],
            "text": m["text"]
        })
    return results

# ---------------------------
# Local RAG with Ollama
# ---------------------------
def rag_answer(query: str, contexts: list[dict], model: str = "llama3.1:8b") -> str:
    """
    If Ollama is installed and running (http://localhost:11434),
    compose an answer using top-k retrieved chunks.
    Ask Ollama with context + return citations.
    """
    try:
        import requests
        sys_prompt = (
            "You are a world class helpful study assistant and mentor. Use ONLY the provided context to answer.\n"
            "Always cite the document filename and page in your answer.\n"
            "If the answer is not in the context, say you don't know.\n"
        )
        ctx = "\n\n".join([f"[{i+1}] {c['text']} (source: {c['filename']} p.{c['page']})" for i, c in enumerate(contexts)])
        prompt = f"{sys_prompt}\nContext:\n{ctx}\n\nQuestion: {query}\nAnswer with citations::"
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60
        )
        if resp.ok:
            return resp.json().get("response", "").strip()
        return "Ollama response error."
    except Exception as e:
        return f"Ollama not available or failed: {e}"


def highlight_text(text: str, query: str) -> str:
    """Highlight query terms inside text (case-insensitive)."""
    # Escape query so regex doesn’t break with special characters
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    highlighted = pattern.sub(lambda m: f"**:orange[{m.group(0)}]**", text)
    return highlighted

# ✅ Safe wrapper for Ollama calls
def safe_rag_answer(prompt, hits=None, model="mistral"):
    try:
        return rag_answer(prompt, hits or [], model=model)
    except Exception as e:
        return f"⚠️ Ollama not available or failed: {e}"

# -----------------------------
# HF Pipelines — Eager loading helpers
# -----------------------------
def _load_seq2seq_pipeline(model_name: str, task: str, device_index: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        device_map=None,          # <- hard disable meta/accelerate sharding
    )
    if device_index >= 0:
        model = model.to("cuda")
    # else CPU by default; do NOT call .to("cpu") to avoid meta-copy errors
    return pipeline(task, model=model, tokenizer=tokenizer, device=device_index)

def _load_qa_pipeline(model_name: str, device_index: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        device_map=None,
    )
    if device_index >= 0:
        model = model.to("cuda")
    return pipeline("question-answering", model=model, tokenizer=tokenizer, device=device_index)

@st.cache_resource(show_spinner=False)
def load_summarizer():
    if torch.cuda.is_available():
        return _load_seq2seq_pipeline("sshleifer/distilbart-cnn-12-6", "summarization", device_index=0)
    else:
        return _load_seq2seq_pipeline("t5-small", "summarization", device_index=-1)

@st.cache_resource(show_spinner=False)
def load_qa_pipeline():
    if torch.cuda.is_available():
        return _load_qa_pipeline("deepset/roberta-base-squad2", device_index=0)
    else:
        return _load_qa_pipeline("distilbert-base-uncased-distilled-squad", device_index=-1)

@st.cache_resource(show_spinner=False)
def load_qg_pipeline():
    if torch.cuda.is_available():
        model_name = "iarfmoose/t5-base-question-generator"
        device = 0
    else:
        model_name = "valhalla/t5-small-qa-qg-hl"
        device = -1

    # ⚡ Force eager load (no meta tensors)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,   # ✅ disable low-mem lazy load
        device_map=None,           # ✅ force single-device load
        local_files_only=False     # ✅ redownload if incomplete
    ).to("cuda" if device == 0 else "cpu")

    # 🚨 Safety check: catch meta tensors early
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            raise RuntimeError(
                f"❌ Model parameter {name} is still on 'meta' device! "
                "Weights were not loaded correctly. Try deleting cache: "
                "`~/.cache/huggingface/transformers` and re-download."
            )

    print(f"[DEBUG] QG model loaded on {next(model.parameters()).device}")

    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)


# ---- Initialize once ----
summarizer = load_summarizer()
qa_pipeline = load_qa_pipeline()
qg_pipeline = load_qg_pipeline()

# ---- Summarization ----
def summarize_text(text: str, max_words=120):
    if not text.strip():
        return "No text to summarize."
    summary = summarizer(
        text,
        max_length=max_words,
        min_length=30,
        do_sample=False
    )
    return summary[0]['summary_text']

# ---- Flashcard Generation ----
def generate_ollama_flashcards(text, model="llama3.1:8b", n_cards=6, start_page=None, end_page=None):
    """
    Calls Ollama (or any backend LLM) to generate flashcards.
    Returns a list of dicts: [{"question": "...", "answer": "..."}]
    """
    prompt = f"""
    You are an AI teacher. Generate {n_cards} study flashcards from this text.

    Output MUST be ONLY valid JSON (no explanations, no markdown).
    Format:
    [
      {{"question": "...", "answer": "..."}},
      {{"question": "...", "answer": "..."}}
    ]

    Text range: pages {start_page} to {end_page} (if provided).
    Text: {text}
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60
        )
        data = response.json()
        raw = data.get("response", "")
    except Exception as e:
        return [{"question": "Ollama error", "answer": str(e)}]

    # --- safe JSON parser ---
    def safe_json_parse(s):
        try:
            return json.loads(s)
        except Exception:
            match = re.search(r"\[.*\]", s, re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return None

    flashcards = safe_json_parse(raw)
    if not flashcards:
        flashcards = [{"question": "Error parsing flashcards", "answer": raw[:300]}]

    return flashcards

def generate_hf_flashcards(text: str, num_q=5):
    """Generate Q&A flashcards using Hugging Face pipelines (QG + QA)."""

    # ensure beams >= return sequences
    num_beams = max(num_q, 6)

    # Step 1: Generate candidate questions
    outputs = qg_pipeline(
        "generate questions: " + text,
        max_length=64,
        num_return_sequences=num_q,
        do_sample=True,
        top_k=50,
        num_beams=num_beams    # ✅ safe
    )
    raw_questions = [o["generated_text"] for o in outputs]

    # Step 2: Clean questions
    questions = [q.strip() for q in raw_questions if len(q.strip()) > 10 and "?" in q]

    # Step 3: Generate answers
    flashcards = []
    for q in questions:
        try:
            ans = qa_pipeline(question=q, context=text[:2000])["answer"]
        except Exception:
            ans = "No clear answer found."

        ans = ans.strip().capitalize()
        if ans and ans.lower() != "no answer":
            flashcards.append({"question": q, "answer": ans})

    # Step 4: Fallback if nothing
    if not flashcards:
        flashcards = [{"question": "What is the main idea of this text?",
                       "answer": summarize_text(text)}]

    return flashcards[:num_q]

# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Semantic Search Engine", page_icon="🔎", layout="wide")
st.title("🔎 Semantic Search Engine for Smarter Learning")

with st.sidebar:
    st.header("📄 Ingest Documents")
    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Build / Update Index") and uploads:
        with st.spinner("Indexing..."):
            added = build_or_update_index(uploads)
        st.success(f"Indexed {added} chunks.")

    if st.button("🔄 Reset Index"):
        if os.path.exists(FAISS_PATH):
            os.remove(FAISS_PATH)
        if os.path.exists(META_PATH):
            os.remove(META_PATH)
        st.success("Index reset. You can now upload a new book.")

    st.divider()
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K results", 3, 20, 5)
    use_ollama = st.checkbox("Generate answer with Ollama (optional)", value=False)
    ollama_model = st.text_input("Ollama model", value="llama3.1:8b", help="E.g., llama3.1:8b, mistral, qwen2.5, etc.")

submit = False
query = ""

mode = st.sidebar.radio("App Mode", ["🔍 Search", "📝 Summarize", "🎓 Flashcards"])

if mode == "🔍 Search":
    query = st.text_input("Ask a question or type keywords", placeholder="e.g., How does PCA reduce dimensions?")
    submit = st.button("Search", key="search_btn")

    if submit and query.strip():
        with st.spinner("Searching..."):
            hits = search(query, k=top_k)
        if not hits:
            st.warning("No index yet or no results. Upload PDFs and build the index from the sidebar.")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("Results")
                for i, h in enumerate(hits, 1):
                    st.markdown(
                        f"**{i}. {h['filename']}** — page {h['page']}  \n"
                        f"*similarity:* `{h['score']:.3f}`"
                    )
                    snippet = h["text"][:1000] + ("…" if len(h["text"]) > 1000 else "")
                    st.write(highlight_text(snippet, query))
                    st.markdown("---")

            with c2:
                if use_ollama:
                    with st.spinner("Asking local LLM (Ollama)…"):
                        answer = rag_answer(query, hits, model=ollama_model)
                    st.subheader("Answer (Local RAG)")
                    st.write(answer)

elif mode == "📝 Summarize":
    doc_choice = None
    index, metadata = load_index()
    if metadata:
        doc_choice = st.selectbox("Select a document", sorted(set(m["filename"] for m in metadata)))
    if st.button("Summarize") and doc_choice:
        with st.spinner("Summarizing..."):
            full_text = " ".join([m["text"] for m in metadata if m["filename"] == doc_choice])

            if use_ollama:
                summary = safe_rag_answer(
                    f"""
                    You are a professional teacher creating a short, clear summary for students.

                    Rules:
                    - Output ONLY plain text (no JSON, no links).
                    - Make it concise (6–8 bullet points max).
                    - Highlight key concepts.
                    - Use bullet points (•) for clarity.

                    Text:
                    {full_text[:5000]}
                    """,
                    [],
                    model=ollama_model
                )
            else:
                chunks = full_text[:1500]
                hf_summary = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
                summary = "• " + hf_summary[0]['summary_text'].replace(". ", "\n• ")

        st.subheader("📄 Summary")
        st.write(summary)

elif mode == "🎓 Flashcards":
    if "flashcards_data" not in st.session_state:
        st.session_state.flashcards_data = []

    index, metadata = load_index()
    doc_choice = None

    if metadata:
        doc_choice = st.selectbox(
            "Select a document",
            sorted(set(m["filename"] for m in metadata))
        )
        if doc_choice:
            pages = [m["page"] for m in metadata if m["filename"] == doc_choice]
            lo, hi = min(pages), max(pages)
            min_page, max_page = st.slider("Select page range", lo, hi, (lo, min(lo+9, hi)))
        else:
            min_page, max_page = 1, 1

    if st.button("Generate Flashcards") and doc_choice:
        with st.spinner("Generating flashcards..."):
            filtered_text = " ".join(
                m["text"] for m in metadata
                if m["filename"] == doc_choice and min_page <= m["page"] <= max_page
            )

            if use_ollama:
                st.session_state.flashcards_data = generate_ollama_flashcards(
                    filtered_text,
                    model=ollama_model,
                    n_cards=6,
                    start_page=min_page,
                    end_page=max_page
                )
            else:
                st.session_state.flashcards_data = generate_hf_flashcards(filtered_text, num_q=6)

    if st.session_state.flashcards_data:
        st.subheader("🎓 Flashcards")
        st.markdown("""
            <style>
            .flip-card {
                background: transparent;
                width: 100%;
                height: 220px;
                perspective: 1000px;
                margin-bottom: 20px;
            }
            .flip-card-inner {
                position: relative;
                width: 100%;
                height: 100%;
                text-align: center;
                transition: transform 0.6s;
                transform-style: preserve-3d;
            }
            .flip-card:hover .flip-card-inner {
                transform: rotateY(180deg);
            }
            .flip-card-front, .flip-card-back {
                position: absolute;
                width: 100%;
                height: 100%;
                -webkit-backface-visibility: hidden;
                backface-visibility: hidden;
                border-radius: 16px;
                box-shadow: 0 6px 14px rgba(0,0,0,.12);
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                padding: 16px;
            }
            .flip-card-front {
    background: linear-gradient(135deg, #404F68, #353A3E);
    color: #f0f0f0; /* light text for contrast */
    transition: background 0.6s ease, transform 0.6s ease;
}

.flip-card-back {
    background: linear-gradient(135deg, #333333, #404F68);
    color: #e0e0e0; /* softer white text */
    transform: rotateY(180deg);
    transition: background 0.6s ease, transform 0.6s ease;
}

.flip-card-inner {
    position: relative;
    width: 100%;
    height: 100%;
    text-align: center;
    transition: transform 0.8s ease-in-out;
    transform-style: preserve-3d;
}

            .q, .a {
                max-width: 92%;
                line-height: 1.35;
            }
            </style>
        """, unsafe_allow_html=True)

        # Show in 3 columns
        cols = st.columns(3)
        for i, card in enumerate(st.session_state.flashcards_data[:9], 1):  # up to 9 cards
            with cols[i % 3]:
                q = card.get("question", "")
                a = card.get("answer", "")
                st.markdown(f"""
                    <div class="flip-card">
                      <div class="flip-card-inner">
                        <div class="flip-card-front">
                          <p class="q"><b>Q:</b> {q}</p>
                        </div>
                        <div class="flip-card-back">
                          <h4>✅ Answer</h4>
                          <p class="a">{a}</p>
                        </div>
                      </div>
                    </div>
                """, unsafe_allow_html=True)