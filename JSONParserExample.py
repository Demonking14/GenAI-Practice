from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = JsonOutputParser()
template = PromptTemplate(
    template='Give me name, age and city of any frictional Character \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.format()
# result = model.invoke(prompt)
# print(result)

# final_result = parser.parse(result.content);
# print(final_result)

# Above code is the normal way to use JSONOutputParser , we can also use Chain 

chain = template | model | parser
result = chain.invoke({})
print(result);
# Drawback of using JSONOutputParser is , it doesn't enfore any schema meaning it decides itself in what format it has to give output . to overcome this we can use  StructuredOutputParser because in this we can provide our custom schemas