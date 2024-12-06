import time
import warnings
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tavily.errors import InvalidAPIKeyError
import streamlit as st

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="FindWise",
    page_icon=":material/search:"
)

st.title("🔍 FindWise")

groq_key = st.sidebar.text_input("Groq API Key", type="password")
tavily_key = st.sidebar.text_input("Tavily API Key", type="password")

if not groq_key or not tavily_key:
    st.warning("Enter Groq and Tavily API keys")

def stream_output(response:str):
    for i in response.split():
        yield i + " "
        time.sleep(0.1)

def search_web(query:str) -> list:
  """Search web"""
  try:
    tavily_tool = TavilySearchAPIRetriever(k=3,
                                          api_key=tavily_key,
                                          max_tokens=10000,
                                          search_depth='advanced')
    results = tavily_tool.invoke(query)
    return [result.page_content for result in results]
  except InvalidAPIKeyError as e:
     return False

if 'history' not in st.session_state:
  st.session_state.history = []

if st.sidebar.button("Clear chat"):
       st.session_state.history = []
       st.rerun()

query = st.chat_input(placeholder="Search")
       
if query:
    try:
      llm = ChatGroq(api_key=groq_key, # type: ignore
                model='llama-3.2-1b-preview',
                temperature=0,
                max_retries=3,
                timeout=None,
                max_tokens=1024)

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
      context = search_web(query)
      if not context:
         st.error("Invalid Tavily API Key")
      else:
        response = chain.invoke({"question":query,
                  "context":context,
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
    except Exception as e:
       st.error(f"Error occured:{e}")
st.sidebar.markdown("---")
st.sidebar.write("By Mudassir Junejo")
st.sidebar.markdown("[GitHub](https://github.com/10mudassir007) | [LinkedIN](https://www.linkedin.com/in/mudassir-junejo/) | [GMail](mailto:muddassir032@gmail.com)")