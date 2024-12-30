import os
import warnings
import faiss
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
from io import BytesIO

st.set_page_config(
    page_title="📄 DocLens",
    page_icon=":material/article:"
)


warnings.filterwarnings('ignore')

st.title("DocLens")

groq_api = st.sidebar.text_input("Groq API Key", type="password")

if not groq_api:
    st.write("")

if groq_api:
    uploaded_file = st.file_uploader("Choose a document", accept_multiple_files=True, type=["pdf", "docx", "txt"])

    if uploaded_file:
        all_docs = []
        for uploaded_file in uploaded_file:
            with BytesIO(uploaded_file.read()) as file_data:
                temp_filename = f"temp_{uploaded_file.name}"
                with open(temp_filename, "wb") as f:
                    f.write(file_data.getvalue())
                loader = PyMuPDFLoader(temp_filename)
                docs = loader.load()
                all_docs.extend(docs)
                os.remove(temp_filename)

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, chunk_overlap=30)
        splitted_docs = splitter.split_documents(all_docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
        if splitted_docs:
            vector_db = FAISS.from_documents(documents=splitted_docs, embedding=embeddings)
            retriever = vector_db.as_retriever(k=2)

            prompt = PromptTemplate(
                template="""You are an assistant for question-answering tasks.
                Use the following documents to answer the question.
                If you don't know the answer, just say that you don't know.
                Use three sentences maximum and keep the answer concise:
                Question: {question}
                Documents: {documents}
                Answer:
                """,
                input_variables=['question', "documents"]
            )

            try:
                llm = ChatGroq(
                    api_key=groq_api,
                    model="llama-3.2-1b-preview",
                    temperature=0,
                    max_tokens=1024,
                    timeout=None,
                    max_retries=2,
                )
                rag_chain = prompt | llm | StrOutputParser()

                def generate(query):
                    try:
                        retrieved_docs = retriever.get_relevant_documents(query=query)
                        retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])
                        response = rag_chain.stream({"question": query, "documents": retrieved_text})
                        return response
                    except Exception as e:
                        return f"Error during response generation: {e}"

                query = st.chat_input("Enter your query:")
                if 'messages' not in st.session_state:
                    st.session_state.messages = []

                if query:
                    st.chat_message("user").markdown(query)
                    st.session_state.messages.append({'role': 'user', 'content': query})

                    response = generate(query)
                    if isinstance(response, str) and "Error" in response:
                        st.chat_message('ai').markdown(f"**Error**: {response}")
                    else:
                        res_str = ' '.join(response)
                        st.chat_message('ai').markdown(res_str)
                    st.session_state.messages.append({'role': 'ai', 'content': response})
                    clear_chat = st.button("Clear Chat")
                else:
                    clear_chat = False
                if clear_chat:
                    st.session_state.messages = []
            except Exception as e:
                st.sidebar.error(f"Error initializing Groq LLM: {e}")
    else:
        st.write("Upload File")
else:
    st.sidebar.warning("Please enter and verify your Groq API Key to continue.")
st.sidebar.markdown("---")
st.sidebar.write("By Mudassir Junejo")
st.sidebar.markdown("[GitHub](https://github.com/10mudassir007) | [LinkedIN](https://www.linkedin.com/in/mudassir-junejo/) | [GMail](mailto:muddassir032@gmail.com)")