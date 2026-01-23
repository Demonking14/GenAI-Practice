from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

template1 = PromptTemplate(
    template='Write a detailed 20 line report on the following topic : {topic}',
    input_variables=['topic']
)
template2 = PromptTemplate(
    template='Write a 5 line on the following topic : {text}',
    input_variables=['text']
)

# prompt1 = template1.invoke({'topic' : 'Black-hole'})
# result = model.invoke(prompt1)

# prompt2=template2.invoke({'text':result.content})

# result2=model.invoke(prompt2)
# print(result2.content)


# Above code is the normal way to do instead of StringOutputParser and below will be the code using strOutputParser

parser = StrOutputParser()

chain = template1 | model | parser |template2 | model |parser
result  = chain.invoke({'topic':'black-hole'});
print(result)