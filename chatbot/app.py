from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

##prompt templet
prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Pleasr response the quary"),
        ("user", "Question: {question}")
    ]
)

##streamlit framework
st.title('Mini chatbot with Groq LLM')
input_text = st.text_input("Enter your question here:")

#openAi framework
llm=ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
output_parser=StrOutputParser()
chain=prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
