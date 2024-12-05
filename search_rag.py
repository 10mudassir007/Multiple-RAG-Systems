import os
import time
from dotenv import load_dotenv
import warnings
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import streamlit as st



load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
warnings.filterwarnings('ignore')

def stream_output(response:str):
    for i in response.split():
        yield i + " "
        time.sleep(0.1)

def search_web(query:str) -> list:
  """Search web"""
  tavily_tool = TavilySearchAPIRetriever(k=3,
                                         api_key=tavily_key,
                                         max_tokens=10000,
                                         search_depth='advanced')
  results = tavily_tool.invoke(query)
  return [result.page_content for result in results]

if 'history' not in st.session_state:
  st.session_state.history = []

if st.sidebar.button("Clear chat"):
       st.session_state.history = []
       st.rerun()

st.title("RAG For Searches")

query = st.chat_input(placeholder="Search")
       
if query:
    llm = ChatGroq(api_key=groq_key, # type: ignore
               model='llama-3.2-1b-preview',
               temperature=0,
               max_retries=3,
               timeout=None,
               max_tokens=1024)

    # prompt = """Answer the question according to the giving context, 
    #         provide accurate response and don't hallucinate.
    #         Give the answer in bullet points each point in a new line but don't mention it in the response.
    #         Also be brief response should be of 5 lines/points max.
            
    #         Question:{question}
    #         Context:{context}
    #         Answer: 
    #      """
    prompt = """You are a helpful assistant. Here is the conversation history so far:
          {conversation_history}
        
          Answer the question according to the query and given context:
          Question: {question}
          Context: {context}
          Provide accurate response in bullet points but don't mention it in the response,
          the answer should be brief 5 lines/points max.
          Do not hallucinate: 
         """
    prompt_template = ChatPromptTemplate.from_template(prompt)

    chain = prompt_template | llm
    conversation_history = ""
    for messages in st.session_state.history:
       conversation_history += f"User:{messages['user']}\nAI:{messages['ai']}\n"

    response = chain.invoke({"question":query,
              "context":search_web(query),
              'conversation_history':conversation_history})
    

    if response:
      st.session_state.history.append({"user":query,'ai':response.content})

      for messages in st.session_state.history[:-1]:
        try:
          st.chat_message('user').markdown(messages['user'])
          st.chat_message('ai').markdown(messages['ai'])
        except:
          continue

      st.chat_message('user').markdown(query)
      st.chat_message('ai').write_stream(stream_output(str(response.content)))
  
    