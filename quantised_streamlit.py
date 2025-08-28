import os
import shutil
import pickle
from typing import List, Union, Optional

import torch
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.memory import ConversationBufferWindowMemory

# >>>----->>> Configuration >>>----->>>
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIR = "faiss_index_global"
ALL_CHUNKS_FILE = "all_chunks.pkl"
LAST_CHUNKS_FILE = "last_chunks.pkl"
MAX_HISTORY = 5
GEN_MODEL = "togethercomputer/LLaMA-2-7B-32K-Instruct"
OFFLOAD_DIR = "model_offload"
RETRIEVE_K = 3

# >>>----->>> Quantization settings >>>----->>>
USE_QUANTIZATION = True
BIT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# >>>----->>> Embedding / Indexing Functions>>>----->>>
def load_documents(file_path: str) -> List[str]:
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    result = converter.convert(file_path)
    markdown = result.document.export_to_markdown()
    return [markdown]


def split_documents(texts: List[str], method: str = "header") -> List[str]:
    raw_chunks: List[Union[str, object]] = []
    if method == "header":
        headers = [("###", "Header 3"), ("##", "Header 2"), ("#", "Header 1")]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    for text in texts:
        raw_chunks.extend(splitter.split_text(text))
    chunks: List[str] = []
    for chunk in raw_chunks:
        chunks.append(chunk.page_content if hasattr(chunk, 'page_content') else str(chunk))
    return chunks


def embed_and_save(chunks: List[str], model_name: str, out_dir: str):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local(out_dir)


def load_faiss_index(out_dir: str, model_name: str):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(out_dir, embeddings, allow_dangerous_deserialization=True)


def retrieve(query: str, db, k: int = RETRIEVE_K) -> List[str]:
    return [doc.page_content for doc in db.similarity_search(query, k=k)]

# >>>----->>> Generation Functions >>>----->>>
def init_model(model_name: str, offload_folder: Optional[str] = OFFLOAD_DIR):
    os.makedirs(offload_folder, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
        "offload_folder": offload_folder
    }
    if USE_QUANTIZATION:
        load_kwargs["quantization_config"] = BIT_CONFIG
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    return gen, tokenizer

class Llama32kGenerator:
    def __init__(self, max_history=MAX_HISTORY, offload_folder: Optional[str] = OFFLOAD_DIR):
        self.gen, self.tokenizer = init_model(GEN_MODEL, offload_folder=offload_folder)
        self.memory = ConversationBufferWindowMemory(k=max_history)

    def generate(self, query: str, retrieved: List[str], tone="neutral", temperature=0.7, max_new_tokens=200):
        context = "\n".join(retrieved[:RETRIEVE_K] + [f"User Query: {query}"])
        history = self.memory.load_memory_variables({}).get("history", "")
        prompt = (
            "[INST]\n"
            f"You are a {tone} assistant. Use the following context to answer concisely.\n\n"
            f"Context:\n{context}\n\n"
            f"Conversation History:\n{history}\n"
            "[/INST]\n"
        )
        resp = self.gen(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )[0]["generated_text"]
        answer = resp.split("[/INST]")[-1].strip()
        self.memory.save_context({"input": query}, {"output": answer})
        return answer




# >>>----->>> Streamlit App >>>----->>>
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot with Streamlit")

st.sidebar.header("🔄 Embedding")
uploaded_files = st.sidebar.file_uploader("Upload PDF/DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
split_method = st.sidebar.selectbox("Split method", ["header", "semantic"])
if st.sidebar.button("Add to Embeddings"):
    all_chunks = pickle.load(open(ALL_CHUNKS_FILE, "rb")) if os.path.exists(ALL_CHUNKS_FILE) else []
    last_chunks = []
    os.makedirs("uploaded_docs", exist_ok=True)
    for uploaded_file in uploaded_files:
        fp = os.path.join("uploaded_docs", uploaded_file.name)
        with open(fp, "wb") as f: f.write(uploaded_file.getbuffer())
        chunks = split_documents(load_documents(fp), method=split_method)
        all_chunks.extend(chunks)
        last_chunks = chunks
    pickle.dump(all_chunks, open(ALL_CHUNKS_FILE, "wb"))
    pickle.dump(last_chunks, open(LAST_CHUNKS_FILE, "wb"))
    embed_and_save(all_chunks, MODEL_NAME, EMBED_DIR)
    embed_and_save(last_chunks, MODEL_NAME, EMBED_DIR + "_last")
    st.sidebar.success("Embedding complete!")

st.header("💬 Chat")
query = st.text_input("Enter your query:")
source_option = st.radio("Query Source", ("All Files", "Last Uploaded File"))
gen = Llama32kGenerator()
if st.button("Ask") and query:
    idx = EMBED_DIR if source_option == "All Files" else EMBED_DIR + "_last"
    if not os.path.exists(idx): st.error("Index not found. Embed first.")
    else:
        db = load_faiss_index(idx, MODEL_NAME)
        retrieved = retrieve(query, db)
        answer = gen.generate(query, retrieved)
        st.subheader("Answer:")
        st.write(answer)
        with st.expander("🔍 Retrieved Chunks"):
            for i, c in enumerate(retrieved[:RETRIEVE_K], 1): st.markdown(f"**Chunk {i}:** {c}")
