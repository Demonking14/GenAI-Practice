from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt
load_dotenv();
model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
)

st.header('Research Tool')
paper_input = st.text_input("Enter paper name")
style_input = st.text_input("Enter the style")
length_input=st.text_input("Enter the length of paragraph")

template = load_prompt('template.json');

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke(
        {
        'paper_input' :paper_input,
        'style_input':style_input,
        'length_input':length_input
}

    )
    st.write(result.content)