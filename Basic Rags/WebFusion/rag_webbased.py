import os 
import time
import warnings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="WebFusion RAG",
    page_icon=":material/find_in_page:"
)

st.title("🌐 WebFusion RAG")

groq_key = st.sidebar.text_input("Groq API Key",type='password')
if not groq_key:
    st.warning("Enter Groq API key")

def retrieve_docs(query : str):
  retrieved_docs = retriever.invoke(query)
  retrieved_text = "\n".join([page.page_content for page in retrieved_docs])
  return retrieved_text

def stream(inputs : str):
    for i in inputs.split():
        yield i + " "
        time.sleep(0.07)

if 'url' not in st.session_state:
    st.session_state.url = ""
url = st.sidebar.text_input("Enter URL")

if st.session_state.url == "" and groq_key:
    if url:
        st.session_state.url = url

if 'history' not in st.session_state:
    st.session_state.history = []
clear_chat = st.sidebar.button("Clear chat")
if clear_chat:
    st.session_state.history = []
    st.session_state.url = ""
    st.rerun()

if st.session_state.url:
    
    loader = WebBaseLoader(web_paths=[st.session_state.url])
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700,
                                            chunk_overlap=30,
                                            separators=['\n','.',',','|'])
    splitted_pages = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')    
    vector_db = FAISS.from_documents(
        embedding=embedding_model,
        documents=splitted_pages
    )
    retriever = vector_db.as_retriever(k=3)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an assistant for question-answering tasks. Use the following text extracted from a webpage to answer the question. Include the user's conversation history for context. If you don't know the answer, just say that you don't know. Keep the answer concise and brief."
        ),
        HumanMessagePromptTemplate.from_template(
            "History: {history}\nQuestion: {question}\nDocuments: {documents}"
        ),
        AIMessagePromptTemplate.from_template(
            "Answer:"
        )
    ])
    try:
        llm = ChatGroq(
            api_key=groq_key,
            model="llama-3.2-1b-preview",
            temperature=0.15,
            max_tokens=1024,
            max_retries=3,
            timeout=None
        )
        
        query = st.chat_input("Enter query")
        if not query:
            st.sidebar.success("WebPage Processed\n Start querying")
        if query:
            chain = prompt | llm | StrOutputParser()

            
            def output(query: str):
                retrieved_text = retrieve_docs(query)
                formatted_history = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in st.session_state.history])
                question_dict = {
                    "history": formatted_history,
                    "question": query,
                    "documents": retrieved_text
                }
                response = chain.invoke(question_dict)
                
                return response       
            
            response = output(query)
            response = str(response).split("User:")[0]

            
            
            st.session_state.history.append((query,response))

            for messages in st.session_state.history[:-1]:
                try:
                    st.chat_message("user").markdown(messages[0])
                    st.chat_message('ai').markdown(messages[1])
                except Exception as e:
                    st.error(f"Error Occured:{e}")
            st.chat_message("user").markdown(query)
            st.chat_message('ai').write_stream(stream(response))
            
    except Exception as e:
        st.error(f"Error occured:{e}")
st.sidebar.markdown("---")
st.sidebar.write("By Mudassir Junejo")
st.sidebar.markdown("[GitHub](https://github.com/10mudassir007) | [LinkedIN](https://www.linkedin.com/in/mudassir-junejo/) | [GMail](mailto:muddassir032@gmail.com)")