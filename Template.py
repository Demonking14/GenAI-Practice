from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
    template=""" Please write the small research paper on  "{paper_input}"with the following  specifications:
    Explanation style:{style_input}
    Explanation Length: {length_input}
    1. Mathematical Details:
    -Include relevant mathematical equations if possible
    -Explain the mathematical concepts using simple , intuitive code snippets where applicable. 
    Ensure the research paper  is clear, accurate, and aligned with the provided style and length.
    """,
    input_variables=['paper_input' , 'style_input' , 'length_input']
)

template.save('template.json');