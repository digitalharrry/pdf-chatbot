import os
import numpy as np
import streamlit as st
from pypdf import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer


# ---------- Groq client ----------
@st.cache_resource
def get_groq_client():
    api_key = None
    # Try Streamlit secrets first
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error(
            "No Groq API key found.\n\n"
            "Set GROQ_API_KEY in environment variables or .streamlit/secrets.toml."
        )
        st.stop()

    return Groq(api_key=api_key)


# ---------- Embedding model (local, no API) ----------
@st.cache_resource
def get_embedder():
    # Small, fast model suitable for CPU
    return SentenceTransformer("all-MiniLM-L6-v2")


client = get_groq_client()
embedder = get_embedder()


# ---------- PDF -> Text ----------
def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text.strip()


# ---------- Text chunking ----------
def chunk_text(text, chunk_size=800, overlap=200):
    """
    Simple word-based chunking.
    chunk_size & overlap are in number of words.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------- Embeddings (local) ----------
def get_embeddings(texts):
    """
    texts: list[str] or str
    returns: np.array of shape (n, dim)
    """
    if isinstance(texts, str):
        texts = [texts]
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return emb.astype("float32")


# ---------- Build knowledge base from PDF ----------
def build_knowledge_base(pdf_file):
    raw_text = extract_text_from_pdf(pdf_file)
    if not raw_text.strip():
        st.error("Could not extract any text from this PDF.")
        st.stop()

    chunks = chunk_text(raw_text, chunk_size=800, overlap=200)
    if not chunks:
        st.error("Chunking failed – no chunks generated from PDF.")
        st.stop()

    embeddings = get_embeddings(chunks)
    kb = {
        "chunks": chunks,
        "embeddings": embeddings,
    }
    return kb


# ---------- Retrieval ----------
def retrieve_relevant_chunks(query, kb, top_k=4):
    query_emb = get_embeddings(query)[0]  # shape (dim,)
    doc_embs = kb["embeddings"]           # shape (n, dim)

    # cosine similarity
    doc_norms = np.linalg.norm(doc_embs, axis=1) + 1e-10
    query_norm = np.linalg.norm(query_emb) + 1e-10
    scores = (doc_embs @ query_emb) / (doc_norms * query_norm)

    top_indices = scores.argsort()[-top_k:][::-1]
    return [kb["chunks"][i] for i in top_indices], scores[top_indices]


# ---------- LLM answer via Groq ----------
from groq import APIStatusError

def answer_question(query, kb):
    relevant_chunks, scores = retrieve_relevant_chunks(query, kb, top_k=4)
    context = "\n\n---\n\n".join(relevant_chunks)

    system_prompt = (
        "You are a helpful assistant that answers questions ONLY using the PDF "
        "content provided in the CONTEXT below.\n"
        "- If the answer is not clearly in the context, say you don't know based on the PDF.\n"
        "- Do NOT use outside knowledge.\n"
        "- Be concise and precise."
    )

    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}"

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        answer = completion.choices[0].message.content
        return answer, relevant_chunks, scores

    except APIStatusError as e:
        # Show something human-friendly in the UI
        st.error(
            f"Groq API error (status {e.status_code}).\n\n"
            "Most common reasons:\n"
            "- Invalid or expired API key\n"
            "- Not enough quota / access for this model\n"
            "- Temporary server issue on Groq side\n\n"
            "If this keeps happening, check your key and Groq dashboard."
        )
        # Also print full error to logs for debugging
        print("Groq APIStatusError:", e.status_code, getattr(e, 'body', None))
        st.stop()


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Groq PDF Chatbot", page_icon="⚡")
st.title("⚡ PDF Chatbot")
st.caption("Ask questions and get answers *only* from your uploaded PDF, using Groq for reasoning.")

# Sidebar: upload & process PDF
st.sidebar.header("1️⃣ Upload & Index PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

process_clicked = st.sidebar.button("Build knowledge base")

if process_clicked:
    if uploaded_pdf is None:
        st.sidebar.error("Please upload a PDF first.")
    else:
        with st.spinner("Reading and indexing PDF (this may take a moment)..."):
            kb = build_knowledge_base(uploaded_pdf)
            st.session_state["kb"] = kb
            st.session_state["chat_history"] = []
        st.sidebar.success("PDF processed! You can now ask questions.")

# If no knowledge base yet, show info & stop
if "kb" not in st.session_state:
    st.info("⬅ Upload a PDF in the sidebar and click **Build knowledge base** to start.")
    st.stop()

# Display chat history
for msg in st.session_state.get("chat_history", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question about your PDF...")

if user_input:
    # Save & display user message
    st.session_state.setdefault("chat_history", [])
    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Bot answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking using your PDF and Groq..."):
            answer, relevant_chunks, scores = answer_question(
                user_input, st.session_state["kb"]
            )
            st.markdown(answer)

            # (Optional) show expandable debug: which chunks were used
            with st.expander("Show relevant PDF chunks (debug)"):
                for i, (c, s) in enumerate(zip(relevant_chunks, scores), start=1):
                    st.write(f"**Chunk {i} (similarity={float(s):.3f})**")
                    st.write(c)
                    st.write("---")

    st.session_state["chat_history"].append({"role": "assistant", "content": answer})

