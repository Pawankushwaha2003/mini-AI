from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

##streamlit framework
st.set_page_config(page_title="Mini Chatbot", page_icon="🤖")
st.title("🤖 Mini Chatbot with Groq LLM")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]
    
##prompt templet
prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Pleasr response the quary"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Question: {question}")
    ]
)



#openAi framework
llm=ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
output_parser=StrOutputParser()
chain=prompt | llm | output_parser

#display previous message
for msg in st.session_state.chat_history:
    if isinstance(msg,HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    else:
        with st.chat_message("assistant"):
            st.write(msg.content)

# Chat input
input_text = st.chat_input("Ask me anything...")

if input_text:

    with st.chat_message("user"):
        st.write(input_text)
        
    response=chain.invoke(
        {
            "question": input_text,
            "chat_history": st.session_state.chat_history
        }
    )

     # Show assistant response
    with st.chat_message("assistant"):
        st.write(response)

    # Save conversation
    st.session_state.chat_history.append(HumanMessage(content=input_text))
    st.session_state.chat_history.append(AIMessage(content=response))
