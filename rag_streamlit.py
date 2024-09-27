import os
import warnings
import faiss
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
warnings.filterwarnings('ignore')
uploaded_file = st.file_uploader("Choose a document", type=["pdf", "docx", "txt"])
@st.cache_resource
def load_model_and_tokenizer(model_name, token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
    return tokenizer, model
#directory = r"F:\Files\tutorials\PDF\New folder"
if uploaded_file is not None:
    token = st.sidebar.text_input("Hugging Face Token")
    model_name = st.sidebar.selectbox("Select Model",["LLAMA 3.2","QWEN 2.5"])
    if token:
        #custom_directory = r"F:\Files\Portfolio\models"
        st.write("Loading Model")
        model_name = "meta-llama/Llama-3.2-1B"
        tokenizer, model = load_model_and_tokenizer(model_name, token)
        st.write("Loading File")
        loader = PyMuPDFLoader(uploaded_file)
    
        # Load the document
        docs = loader.load()
    
        # Optionally display some information about the document
        st.write(f"Loaded {len(docs)} document chunks.")
        
        # Display the contents of the loaded document
        st.write("Splitting")
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000,chunk_overlap=15)
        splitted_docs = splitter.split_documents(docs)
    
        st.write("Embedding")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vector_db = FAISS.from_documents(documents=splitted_docs,embedding=embeddings)
        
        retriver = vector_db.as_retriever(k=2)
        
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks.
            Use the following documents to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise:
            Question: {question}
            Documents: {documents}
            Answer:
            """,
            input_variables=['question',"documents"]
        )
        st.write("LLM")
        llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=30, temperature=0.12)
        rag_chain = prompt | llm | StrOutputParser()
        
        
        
        st.title("RAG based Assistance bot")
        query = st.chat_input("Enter your query:")
        if 'messages' not in st.session_state:
                st.session_state.messages = []
        if query:
               st.chat_message("user").markdown(query)
               retrieved_docs = retriver.get_relevant_documents(query=query)
               retreived_text = "\n".join([doc.page_content for doc in retrieved_docs])
               response = rag_chain.invoke({"question":query,"documents":retreived_text})
               generated_prompt = prompt.format({"question": ques, "documents": retreived_text})
        
               llm_response = llm(generated_prompt)[0]['generated_text']
               st.write(llm_response)
               st.write(generated_prompt)
               st.session_state.messages.append({'role':'user','content':query})
               st.chat_message('ai').markdown(response)
               st.session_state.messages.append({'role':'ai','content':response})
               clear_chat = st.button("Clear Chat")
        else:
           clear_chat = False
        if clear_chat:
              st.session_state.messages = []
    else:
        st.write("Enter Token")
else:
    st.write("Upload File")
