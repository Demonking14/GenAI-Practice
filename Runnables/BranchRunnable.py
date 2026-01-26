from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence , RunnableBranch
from pydantic import BaseModel 
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
parser = StrOutputParser()
class Emotions(BaseModel):
    emotion : Literal['Positive' , 'Negative']

EmotionParser = PydanticOutputParser(pydantic_object=Emotions)
    

prompt1 = PromptTemplate(
    template='Tell the emotion of the following review {review}\n {format_instruction}',
    input_variables=['review'],
    partial_variables={'format_instruction' : EmotionParser.get_format_instructions()}

)

prompt2 = PromptTemplate(
    template='Write a appropriate 1 line response for the positive review {review} . Make sure to give only 1 response no options nothing choose yourself',
    input_variables=['review']
)
prompt3 = PromptTemplate(
    template='Write a appropriate 1 line response for the negative review {review} . Make sure to give only 1 response no options nothing choose yourself',
    input_variables=['review']
)

EmotionChain = RunnableSequence(prompt1 , model , EmotionParser)
ResponseChain = RunnableBranch(
    (lambda x : x.emotion== 'Positive' , RunnableSequence(prompt2 , model , parser)),
    (lambda x : x.emotion== 'Negative' , RunnableSequence(prompt3, model , parser)),
    RunnableSequence(PromptTemplate(template="Thank you very much", input_variables=[]), model, parser)
)

result_chain = EmotionChain | ResponseChain
print(result_chain.invoke({'review' :'Pretty bad product but there were so many scratches but product good but also lags a bit'}))