from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# ---------- Setup ----------
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found in .env")
    st.stop()

st.title("📘 Personal Study Helper (PDF RAG)")

# ---------- LLM / Embeddings / Prompt ----------

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

prompt = PromptTemplate(
    template="""
You are a helpful assistant answering questions strictly from the PDF context.

Chat History:
{history}

Context:
{context}

Question:
{question}

Answer ONLY from the context. Mention page numbers if possible.
""",
    input_variables=["history", "context", "question"],
)

# ---------- Cache PDF ----------

@st.cache_resource
def load_pdf(path):
    loader = PyPDFLoader(file_path=path)
    return loader.load()

# ---------- Cache Vector Store ----------

@st.cache_resource
def build_vectorstore(docs, _embeddings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    documents = splitter.split_documents(docs)
    return FAISS.from_documents(documents, _embeddings)

# ---------- Initialize Once ----------

filepath = "damn.pdf"
docs = load_pdf(filepath)

if "retriever" not in st.session_state:
    with st.spinner("Indexing PDF for the first time..."):
        vector_store = build_vectorstore(docs, embeddings)
        st.session_state.retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20},
        )

# ---------- Chat Memory ----------

if "messages" not in st.session_state:
    st.session_state.messages = []

def format_docs(docs):
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def get_history():
    return "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in st.session_state.messages
    )

def ask_question(question):
    retrieved_docs = st.session_state.retriever.invoke(question)
    context = format_docs(retrieved_docs)

    chain = prompt | llm | parser
    return chain.invoke({
        "history": get_history(),
        "context": context,
        "question": question,
    })

# ---------- UI ----------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_question(user_input)
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
