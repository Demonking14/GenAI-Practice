from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence , RunnablePassthrough, RunnableParallel
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()
prompt1 = PromptTemplate(
    template='Generate me a joke on the following topic {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Explain the following joke in 2 line {joke}',
    input_variables=['joke']
)

chain = RunnableSequence(prompt1 , model , parser)
chain2 = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation' : RunnableSequence(prompt2 , model , parser)
}
)
result = RunnableSequence(chain , chain2)
print(result.invoke({'topic':'AI'}))
