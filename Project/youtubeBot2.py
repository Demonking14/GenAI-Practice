from  langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import yt_dlp
import requests
import streamlit as st
load_dotenv()
transcript=""
def get_transcript(url):
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'quiet': True,
        'force_ipv4':True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        subs = info.get("requested_subtitles")
        if not subs or "en" not in subs:
            return None
        sub_url = subs["en"]["url"]
        data = requests.get(sub_url).text
        lines = [line for line in data.split("\n") if "-->" not in line and line.strip()]
        return " ".join(lines)
    
@st.cache_resource
def build_vectorstore(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([transcript])
    embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

prompt = PromptTemplate(
    template="""
You are a helpful assistant answering questions from a YouTube video transcript.

Chat History:
{history}

Context:
{context}

Question: {question}
Answer only from context.
""",
    input_variables=["history", "context", "question"],
)
st.title(" YouTube Video Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

url = st.text_input("Enter YouTube URL")

if url and "retriever" not in st.session_state:
    with st.spinner("Fetching transcript..."):
        transcript = get_transcript(url)
        vector_store = build_vectorstore(transcript)
        st.session_state.retriever = vector_store.as_retriever()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_history():
    return "\n".join(
        [f"User: {m['content']}" if m["role"] == "user" else f"Assistant: {m['content']}"
         for m in st.session_state.messages]
    )

def ask_question(question):
    retriever = st.session_state.retriever
    docs = retriever.invoke(question)
    context = format_docs(docs)

    chain = prompt | llm | parser

    return chain.invoke({
        "history": get_history(),
        "context": context,
        "question": question
    })
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask about the video"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_question(user_input)
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
