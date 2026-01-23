from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import ResponseSchema, StructuredOutputParser
load_dotenv()
# All this will not work because StructuredOutputParser has been deprecated from the new langchain version so learn pydantic and use that 
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
schema = [
    ResponseSchema(name='fact1' , description='fact1 about the topic'),
    ResponseSchema(name='fact2' , description='fact2 about the topic'),
    ResponseSchema(name='fact3' , description='fact3 about the topic'),
    ResponseSchema(name='fact4' , description='fact4 about the topic'),
]
parser = StructuredOutputParser.from_response_schema(schema)


template = PromptTemplate(
    template='Give me 4 facts about the topic {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


chain = template | model |parser
result = chain.invoke({'topic':'blackhole'})
print(result)


