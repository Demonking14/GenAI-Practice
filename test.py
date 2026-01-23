from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv();

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)
messages = [
    (
        "system",
        "You are a philosopher who converts any sentence into a life lesson.",
    ),
    ("human", "I am eating"),
]
ai_msg = model.invoke(messages)
print(ai_msg.content)