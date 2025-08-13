"""
app.py - Streamlit AI Long Text Summarizer
Requirements (system):
  - Tesseract OCR engine must be installed and on PATH:
      - Linux: `sudo apt install tesseract-ocr`
      - macOS: `brew install tesseract`
      - Windows: install from https://github.com/tesseract-ocr/tesseract and add to PATH
  - Poppler (for pdf2image) must be installed:
      - Linux: `sudo apt install poppler-utils`
      - macOS: `brew install poppler`
      - Windows: install poppler binaries and add to PATH
"""

import re
from io import BytesIO

import chardet
import magic                  # python-magic (libmagic wrapper)
import pdfplumber             # pdf text extraction
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
    # deterministic output - set temperature to 0
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key, model = "gpt-4o-mini")
    return llm

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Long Text Summarizer", layout="centered")
st.title("AI Long Text Summarizer")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(
        "Upload TXT / PDF / DOCX files and get an LLM-powered summary. "
        "PDFs that are scanned (images) will be OCR'd automatically."
    )
with col2:
    st.markdown("Contact: [William Sun](mailto:omniai.labs4ever@gmail.com)")

# API key input
st.markdown("## OpenAI API Key")
openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", key="openai_api_key_input")

# File uploader
st.markdown("## Upload a file to summarize")
uploaded_file = st.file_uploader("Choose a file (TXT / PDF / DOCX)", type=["txt", "pdf", "docx"])

# ---------- Main processing ----------
def detect_mime(bytes_data: bytes) -> str:
    try:
        return magic.from_buffer(bytes_data, mime=True)
    except Exception:
        # fallback: basic heuristics from header
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

def extract_text_from_pdf(bytes_data: bytes, ocr_if_needed: bool = True, ocr_dpi: int = 300) -> (str, bool):
    """
    Returns (extracted_text, used_ocr_flag).
    Uses pdfplumber for native text extraction. If that yields very little text
    (likely scanned PDF) and ocr_if_needed=True, performs OCR via pdf2image+pytesseract.
    """
    text_parts = []
    used_ocr = False

    try:
        with pdfplumber.open(BytesIO(bytes_data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                # pdfplumber also supports extracting with advanced layout if needed
                text_parts.append(page_text)
    except Exception as e:
        # fail-safe: continue to OCR path if permitted
        text_parts = []

    raw_text = "\n".join([p for p in text_parts if p is not None])
    # Heuristic: if extracted text is too short, consider OCR fallback
    if ocr_if_needed and (len(raw_text.strip()) < 200):
        try:
            used_ocr = True
            # convert pages to PIL images
            images = pdf2image.convert_from_bytes(bytes_data, dpi=ocr_dpi)
            ocr_texts = []
            for img in images:
                ocr_page = pytesseract.image_to_string(img)
                ocr_texts.append(ocr_page)
            raw_text = "\n".join(ocr_texts)
        except Exception as e:
            # If OCR fails, return whatever text we have (possibly empty)
            used_ocr = False

    return raw_text, used_ocr

def extract_text_from_docx(bytes_data: bytes) -> str:
    # python-docx accepts a file-like BytesIO
    doc = docx.Document(BytesIO(bytes_data))
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)

# ---------- Run when a file is uploaded ----------
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    st.info(f"Uploaded file: {uploaded_file.name} — size: {len(bytes_data)/1024:.1f} KB")

    mime_type = detect_mime(bytes_data)
    st.write(f"Detected MIME type: `{mime_type}`")

    file_input = ""
    used_ocr_flag = False

    # TXT-like
    if mime_type in ("text/plain", "text/utf-8", "text/markdown"):
        file_input = extract_text_from_txt(bytes_data)

    # PDF
    elif mime_type == "application/pdf":
        with st.spinner("Extracting text from PDF (pdfplumber)..."):
            file_input, used_ocr_flag = extract_text_from_pdf(bytes_data, ocr_if_needed=True)
        if used_ocr_flag:
            st.success("OCR fallback used (Tesseract).")
        else:
            st.success("PDF text extracted (no OCR needed).")

    # DOCX
    elif mime_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        with st.spinner("Extracting text from DOCX..."):
            file_input = extract_text_from_docx(bytes_data)
        st.success("DOCX text extracted.")

    else:
        st.error(f"Unsupported or unrecognized file type: {mime_type}")
        st.stop()

    # Basic cleanup
    file_input = file_input.strip()
    if not file_input:
        st.error("No text could be extracted from the uploaded file.")
        st.stop()

    # Word count (use regex for better accuracy)
    word_count = len(re.findall(r"\w+", file_input))
    st.write(f"Word count (approx): {word_count:,}")

    # Guard rails
    MAX_WORDS = 20000
    if word_count > MAX_WORDS:
        st.warning(f"File too long ({word_count} words). Maximum supported: {MAX_WORDS} words.")
        st.stop()

    if not openai_api_key:
        st.warning(
            "OpenAI API key is required to run the summarizer. "
            "Enter it above and re-run.",
            icon="⚠️",
        )
        st.stop()

    # ---------- Prepare for summarization ----------
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=350
    )
    with st.spinner("Splitting text into chunks..."):
        splitted_documents = text_splitter.create_documents([file_input])

    st.write(f"Number of chunks: {len(splitted_documents)}")

    # ---------- Run summarization chain ----------
    llm = load_LLM(openai_api_key=openai_api_key)
    summarize_chain = load_summarize_chain(llm=llm, chain_type="map_reduce")

    with st.spinner("Generating summary (this may take a while depending on chunk count)..."):
        summary_output = summarize_chain.run(splitted_documents)

    st.markdown("### Summary")
    st.write(summary_output)

    # Optionally: show whether OCR was used
    if used_ocr_flag:
        st.info("Note: OCR (Tesseract) was used to extract text from some pages.")
