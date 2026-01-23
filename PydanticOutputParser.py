from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field ,BaseModel
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
class Person(BaseModel):
    name:str = Field(description='Name of a person')
    age:int = Field(gt=18 , description='Age of a person')
    city:str = Field(description='City where person live')

parser = PydanticOutputParser(pydantic_object=Person)
template = PromptTemplate(
    template='Give me name, age and city of any frictional person from {place} \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'place': 'India'})
print(result)