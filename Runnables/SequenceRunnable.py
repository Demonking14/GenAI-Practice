from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
load_dotenv()
model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write few lines about the topic {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Write 1 line summary from the following paragraph {paragraph}',
    input_variables=['paragraph']
)

sequence_runnable = RunnableSequence(prompt , model, parser , prompt2 , model ,parser)
print(sequence_runnable.invoke({'topic': 'Universe'}))