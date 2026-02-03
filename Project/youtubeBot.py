from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled , NoTranscriptFound
from  langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel ,RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

video_id = "3E8IGy6I9Wo"
cookie_path ="www.youtube.com_cookies.txt"
transcript=""
try:
  ytt_api = YouTubeTranscriptApi()
  transcript_list = ytt_api.fetch(video_id=video_id ,languages=["en"])
  transcript= " ".join(chunk.text for chunk in  transcript_list)
#   print(transcript)
except Exception as e:
    print(f"Failed to load transcript:{e}")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=200)
chunks = splitter.create_documents([transcript])
embeddings = GoogleGenerativeAIEmbeddings(
    model='models/gemini-embedding-001'
)
vector_store=FAISS.from_documents(chunks , embeddings)
retriver = vector_store.as_retriever(search_type="similarity" , search_kwargs={"k":4})

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash' )
prompt = PromptTemplate(
    template="""
    You are a helpful assistant. Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know .
    {context}
    Question:{question}""",
    input_types=['context' , 'question']
)

def format_docs(retrieved_docs):
    context_text="\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain= RunnableParallel({
    'context':retriver | RunnableLambda(format_docs),
    'question':RunnablePassthrough()
})
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm |parser

print(main_chain.invoke("Can you summarize the video"))