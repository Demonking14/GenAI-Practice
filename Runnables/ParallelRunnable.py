from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()
prompt1 = PromptTemplate(
    template='write me  a  1 line tweet for the topic {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='write me  a 3 line linkedinPost about the following topic{topic} ',
    input_variables=['topic']
)
parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1 , model , parser),
    'linkedin':RunnableSequence(prompt2 , model , parser)

}
    
)
print(parallel_chain.invoke({'topic':'ChatApp website'}))