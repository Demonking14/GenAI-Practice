from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

prompt = PromptTemplate(
    template='Write 5 Points about the following topic {topic}',
    input_variables={'topic'}

)
parser = StrOutputParser()

chain = prompt | model | parser
print(chain.invoke({'topic':'cricket'}))

# Below line is to visualize the chain 
print(chain.get_graph().print_ascii())