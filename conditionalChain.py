from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel , Field
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch , RunnableLambda
from typing import Literal
from langchain_core.prompts import PromptTemplate
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
class Feedback(BaseModel):
    sentiment : Literal['positive' , 'negative'] = Field(description='Sentiment of the given feedback')

class FeedbackResponse(BaseModel):
    feebackResponse : str = Field(description='2 line response for the following feeeback')


parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)
parser3 = PydanticOutputParser(pydantic_object=FeedbackResponse)

prompt1 = PromptTemplate(
    template='Give the sentiment of the feedback in postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction' : parser2.get_format_instructions()}

)
sentiment_chain = prompt1 | model | parser2
# print(sentiment_chain.invoke({'feedback' : 'The product was broken and in a very bad condition'}))

prompt2 = PromptTemplate(
    template='write  the appropriate 2 line response to this  positive  feedback \n {feedback} \n{format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction' : parser3.get_format_instructions()}
)
prompt3= PromptTemplate(
    template='Write the appropriate 2 line response to this  negative  feedback \n {feedback}',
    input_variables=['feedback'],
    partial_variables={'format_instruction' : parser3.get_format_instructions()}
)

condition_brach = RunnableBranch(
    (lambda x : x.sentiment == 'positive' , prompt2 | model |parser3 ),
    (lambda x : x.sentiment == 'negative' , prompt3 | model |parser3),
    RunnableLambda(lambda x :'Could not fine the feedback')
)

chain = sentiment_chain | condition_brach
print(chain.invoke({'feedback' : 'Best product i have ever purchased'} ).feedbackResponse);

