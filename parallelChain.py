from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Define response type classification
class ResponseType(BaseModel):
    response_type: Literal['notes', 'quiz', 'summary'] = Field(
        description='Type of response to generate: notes, quiz, or summary'
    )

# Parser for response type classification
type_parser = PydanticOutputParser(pydantic_object=ResponseType)

# Prompt to classify what type of response to give
type_prompt = PromptTemplate(
    template='''Analyze this topic and decide what type of educational content would be most appropriate.
    Choose from: 'notes' (detailed explanation), 'quiz' (test questions), or 'summary' (brief overview).

    Topic: {topic}

    Consider:
    - If the topic is complex or new, choose 'notes'
    - If the topic seems like something to test knowledge on, choose 'quiz'
    - If the topic is well-known or simple, choose 'summary'

    {format_instructions}''',
    input_variables=['topic'],
    partial_variables={'format_instructions': type_parser.get_format_instructions()}
)

# Content generation prompts
notes_prompt = PromptTemplate(
    template='Generate detailed 10-line notes on the topic: {topic}',
    input_variables=['topic']
)

quiz_prompt = PromptTemplate(
    template='Generate a 4-question quiz on the topic: {topic}',
    input_variables=['topic']
)

summary_prompt = PromptTemplate(
    template='Provide a brief 3-sentence summary of the topic: {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

# Classification chain
type_chain = type_prompt | model | type_parser

# Content generation chains
notes_chain = notes_prompt | model | parser
quiz_chain = quiz_prompt | model | parser
summary_chain = summary_prompt | model | parser

# Routing logic based on classification
def route_by_type(result):
    if result.response_type == 'notes':
        return notes_chain
    elif result.response_type == 'quiz':
        return quiz_chain
    else:
        return summary_chain

# Create routing chain
routing_chain = RunnableBranch(
    (lambda x: x.response_type == 'notes', notes_chain),
    (lambda x: x.response_type == 'quiz', quiz_chain),
    summary_chain  # default case
)

# Final chain: classify -> route to appropriate content generator
chain = type_chain | routing_chain

# Test the chain
print(chain.invoke({'topic': 'White moon'}))
print('\n' + '='*50 + '\n')

# Test with different topics to see different responses
print("Testing with different topics:")
print("\n1. Complex topic (should give notes):")
print(chain.invoke({'topic': 'Quantum Physics'}))

print("\n2. Test topic (should give quiz):")
print(chain.invoke({'topic': 'World War II'}))

