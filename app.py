import os
import tempfile
import shutil

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Chat", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = None

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-2506")


def build_vectorstore(pdf_path: str, persist_dir: str):
    """Load a PDF, chunk it, embed it, and store it in a fresh Chroma DB."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding_model = MistralAIEmbeddings(model="mistral-embed")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    return vectorstore, len(chunks)


def answer_question(vectorstore, question: str) -> str:
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = PROMPT.invoke({"context": context, "question": question})
    response = get_llm().invoke(final_prompt)
    return response.content


# ---------------------------------------------------------------------------
# Sidebar: PDF upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Only reprocess if it's a new/different file
        if st.session_state.processed_file != uploaded_file.name:
            with st.spinner("Reading, chunking, and embedding your PDF..."):
                # Clean up any previous persist directory
                if st.session_state.persist_dir and os.path.exists(
                    st.session_state.persist_dir
                ):
                    shutil.rmtree(st.session_state.persist_dir, ignore_errors=True)

                # Save uploaded file to a temp path
                tmp_dir = tempfile.mkdtemp()
                pdf_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                persist_dir = os.path.join(tmp_dir, "chroma_db")

                try:
                    vectorstore, num_chunks = build_vectorstore(pdf_path, persist_dir)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.persist_dir = persist_dir
                    st.session_state.messages = []  # reset chat for new doc
                    st.success(f"Processed '{uploaded_file.name}' into {num_chunks} chunks ✅")
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")
        else:
            st.info(f"'{uploaded_file.name}' is already loaded.")

    if st.session_state.vectorstore is not None:
        st.divider()
        if st.button("🗑️ Clear document & chat"):
            if st.session_state.persist_dir and os.path.exists(
                st.session_state.persist_dir
            ):
                shutil.rmtree(st.session_state.persist_dir, ignore_errors=True)
            st.session_state.vectorstore = None
            st.session_state.processed_file = None
            st.session_state.persist_dir = None
            st.session_state.messages = []
            st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("💬 Chat with your PDF")

if st.session_state.vectorstore is None:
    st.info("Upload a PDF from the sidebar to get started.")
else:
    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_query = st.chat_input("Ask a question about your document...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = answer_question(st.session_state.vectorstore, user_query)
                except Exception as e:
                    answer = f"⚠️ Error while generating a response: {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
