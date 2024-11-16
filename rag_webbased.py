import os 
from urllib import response
import warnings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_groq import ChatGroq
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
warnings.filterwarnings('ignore')

def retrieve_docs(query : str):
  retrieved_docs = retriever.invoke(query)
  retrieved_text = "\n".join([page.page_content for page in retrieved_docs])
  return retrieved_text


#https://medium.com/@myscale/enhancing-advanced-rag-systems-using-reranking-with-langchain-523a0b840311
url = st.text_input("Enter url")
url = "https://medium.com/@myscale/enhancing-advanced-rag-systems-using-reranking-with-langchain-523a0b840311"
if url:
    if 'url' not in st.session_state:
        st.session_state.url = url
    st.rerun()
    st.write(url)
    st.write(st.session_state)
    loader = WebBaseLoader(web_paths=[st.session_state.url])
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
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an assistant for question-answering tasks. Use the following text extracted from a webpage to answer the question. Include the user's conversation history for context. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise and brief."
        ),
        HumanMessagePromptTemplate.from_template(
            "History: {history}\nQuestion: {question}\nDocuments: {documents}"
        ),
        AIMessagePromptTemplate.from_template(
            "Answer:"
        )
    ])
    llm = ChatGroq(
        api_key=groq_key,
        model="llama-3.2-1b-preview",
        temperature=0.15,
        max_tokens=1024,
        max_retries=3,
        timeout=None
    )

    query = st.text_input("Enter query")
    if query:
        retrieved_docs = retrieve_docs(query)
        question_dict = {"question":query, "documents":retrieved_docs}
        chain = prompt | llm | StrOutputParser()
        
        def output(query: str):
            if 'history' not in st.session_state:
                st.session_state.history = []
            retrieved_text = retrieve_docs(query)
            formatted_history = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in st.session_state.history])
            question_dict = {
                "history": formatted_history,
                "question": query,
                "documents": retrieved_text
            }
            response = chain.invoke(question_dict)
            st.session_state.history.append((query,response))
            return response       
        
        response = output(query)
        if 'display' not in st.session_state:
            st.session_state.display = []

        for i in history:
            st.session_state.display.append({"query":i[0], "reply":str(i[-1]).split("**Answer:**")[-1]})


        # for messages in st.session_state.display:
        #     st.chat_message("user").markdown(messages['queries'])
        #     st.chat_message('ai').markdown(messages['replies'])
        st.write(st.session_state)
