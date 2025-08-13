"""
app.py - Streamlit AI Long Text Summarizer (Updated for LangChain 0.1+)
"""

import re
import time
from io import BytesIO

import chardet
import magic                  # python-magic (libmagic wrapper)
import pdfplumber             # PDF text extraction
import docx                   # python-docx
import pdf2image              # convert PDF pages to images (for OCR)
import pytesseract            # Tesseract OCR
import streamlit as st
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------- Helper: LLM loader ----------
def load_LLM(openai_api_key: str):
    return OpenAI(temperature=0, openai_api_key=openai_api_key, model="gpt-4o-mini")

# ---------- Retry wrapper for LLM calls ----------
def safe_invoke(chain, docs, retries=3, delay=3):
    last_error = None
    for attempt in range(retries):
        try:
            return chain.invoke({"input_documents": docs})
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise last_error

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Long Text Summarizer", layout="centered")
st.title("AI Long Text Summarizer")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(
        "Upload TXT / PDF / DOCX files and get an LLM-powered summary. "
        "Scanned PDFs will be OCR'd automatically."
    )
with col2:
    st.markdown("Contact: [William Sun](mailto:omniai.labs4ever@gmail.com)")

# API key input
st.markdown("## OpenAI API Key")
openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

# File uploader
st.markdown("## Upload a file to summarize")
uploaded_file = st.file_uploader("Choose a file (TXT / PDF / DOCX)", type=["txt", "pdf", "docx"])

# ---------- Helper functions ----------
def detect_mime(bytes_data: bytes) -> str:
    try:
        return magic.from_buffer(bytes_data, mime=True)
    except Exception:
        head = bytes_data[:20].lower()
        if head.startswith(b'%pdf'):
            return "application/pdf"
        return "application/octet-stream"

def extract_text_from_txt(bytes_data: bytes) -> str:
    guess = chardet.detect(bytes_data).get("encoding") or "utf-8"
    try:
        return bytes_data.decode(guess)
    except Exception:
        return bytes_data.decode("utf-8", errors="ignore")

def extract_text_from_pdf(bytes_data: bytes, ocr_if_needed=True, ocr_dpi=300) -> (str, bool):
    text_parts = []
    used_ocr = False
    try:
        with pdfplumber.open(BytesIO(bytes_data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
    except:
        text_parts = []

    raw_text = "\n".join([p for p in text_parts if p])
    if ocr_if_needed and len(raw_text.strip()) < 200:
        try:
            used_ocr = True
            images = pdf2image.convert_from_bytes(bytes_data, dpi=ocr_dpi)
            ocr_texts = [pytesseract.image_to_string(img) for img in images]
            raw_text = "\n".join(ocr_texts)
        except:
            used_ocr = False
    return raw_text, used_ocr

def extract_text_from_docx(bytes_data: bytes) -> str:
    doc = docx.Document(BytesIO(bytes_data))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# ---------- Main workflow ----------
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    st.info(f"Uploaded file: {uploaded_file.name} — size: {len(bytes_data)/1024:.1f} KB")

    mime_type = detect_mime(bytes_data)
    st.write(f"Detected MIME type: `{mime_type}`")

    file_input, used_ocr_flag = "", False
    if mime_type in ("text/plain", "text/utf-8", "text/markdown"):
        file_input = extract_text_from_txt(bytes_data)
    elif mime_type == "application/pdf":
        with st.spinner("Extracting text from PDF..."):
            file_input, used_ocr_flag = extract_text_from_pdf(bytes_data)
        st.success("OCR fallback used." if used_ocr_flag else "PDF text extracted.")
    elif mime_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"):
        with st.spinner("Extracting text from DOCX..."):
            file_input = extract_text_from_docx(bytes_data)
        st.success("DOCX text extracted.")
    else:
        st.error(f"Unsupported file type: {mime_type}")
        st.stop()

    file_input = file_input.strip()
    if not file_input:
        st.error("No text extracted from file.")
        st.stop()

    word_count = len(re.findall(r"\w+", file_input))
    st.write(f"Word count: {word_count:,}")

    if word_count > 20000:
        st.warning(f"File too long ({word_count} words). Max: 20,000.")
        st.stop()

    if not openai_api_key:
        st.warning("Enter OpenAI API key to summarize.", icon="⚠️")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=350
    )
    with st.spinner("Splitting text into chunks..."):
        splitted_documents = text_splitter.create_documents([file_input])

    st.write(f"Number of chunks: {len(splitted_documents)}")

    llm = load_LLM(openai_api_key)
    summarize_chain = load_summarize_chain(llm=llm, chain_type="map_reduce")

    with st.spinner("Generating summary..."):
        summary_output = safe_invoke(summarize_chain, splitted_documents)

    st.markdown("### Summary")
    st.write(summary_output)

    if used_ocr_flag:
        st.info("OCR (Tesseract) was used for some pages.")
