from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

os.environ["USER_AGENT"] = "MyWebScraper/1.0"

# Now proceed with your LangChain imports and logic
from langchain_community.document_loaders import WebBaseLoader
load_dotenv()
Prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text if data not available then do some calculation . - \n {text}',
    input_variables=['question' , 'text']
    
)
model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash'
)
parser = StrOutputParser()
url = 'https://leetcode.com/problems/longest-common-prefix/'
loader = WebBaseLoader([url])
docs = loader.load()
chain = Prompt | model | parser
print(chain.invoke({'question':'Solve the following question' , 'text':docs[0].page_content}))