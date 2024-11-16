import os 
import warnings
import faiss
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_groq import ChatGroq
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
warnings.filterwarnings('ignore')
#https://medium.com/@myscale/enhancing-advanced-rag-systems-using-reranking-with-langchain-523a0b840311
url = st.text_input("Enter url")
url = "https://medium.com/@myscale/enhancing-advanced-rag-systems-using-reranking-with-langchain-523a0b840311"
if url:
    if 'url' not in st.session_state:
        st.session_state.url = url
    loader = WebBaseLoader(web_paths=[url])
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700,
                                            chunk_overlap=30,
                                            separators=['\n','.',',','|'])
    splitted_pages = splitter.split_documents(documents)


    embedding_model = FastEmbedEmbeddings(
        model_name='BAAI/bge-small-en-v1.5',
        threads=-1,
        batch_size=128,

    )

    vector_db = FAISS.from_documents(
        embedding=embedding_model,
        documents=splitted_pages,
        docstore=InMemoryDocstore()
    )

    retriever = vector_db.as_retriever(k=3)
    st.write(retriever.invoke("What is RAG?"))


    
    
