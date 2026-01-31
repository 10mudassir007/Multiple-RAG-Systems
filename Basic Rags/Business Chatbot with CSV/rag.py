# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langsmith import Client
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import warnings 

# warnings.filterwarnings('ignore')
# load_dotenv()

# # Load first CSV
# loader1 = CSVLoader("employees.csv")
# docs1 = loader1.load()

# # Load second CSV
# loader2 = CSVLoader("revenue.csv")
# docs2 = loader2.load()

# # Combine documents
# docs = docs1 + docs2

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# texts = text_splitter.split_documents(docs)

# embeddings = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-small-english-r2")

# client = QdrantClient(":memory:")

# client.create_collection(
#     collection_name="demo_collection",
#     vectors_config=VectorParams(size=384, distance=Distance.COSINE)
# )

# vector_store = QdrantVectorStore(
#     client=client,
#     collection_name="demo_collection",
#     embedding=embeddings,
# )
# vector_store.add_documents(docs)

# retriever = vector_store.as_retriever()

# # Create a LangSmith API in Settings > API Keys
# # Make sure API key env var is set:
# # import os; os.environ["LANGSMITH_API_KEY"] = "<your-api-key>"

# client = Client()
# prompt = client.pull_prompt("rlm/rag-prompt")

# llm = ChatGroq(
                    
#                     model="openai/gpt-oss-120b",
#                     temperature=0,
#                     max_tokens=1024,
#                     timeout=None,
#                     max_retries=2,
#                 )
                
# rag_chain = prompt |llm 

# while True:
#     x = input("Enter: ")
#     reply = rag_chain.invoke({"question":x,"context":retriever.invoke(x)}).content
#     print(reply)

import streamlit as st
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import warnings
import os

warnings.filterwarnings('ignore')
load_dotenv(override=True)

st.set_page_config(
    page_title="AI Business Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Business Chatbot")

# --- Load CSVs ---
loader1 = CSVLoader("employees.csv")
docs1 = loader1.load()

loader2 = CSVLoader("revenue.csv")
docs2 = loader2.load()

docs = docs1 + docs2

# --- Split documents ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_documents(docs)

# --- Embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-small-english-r2")

# --- Qdrant Vector Store ---
client_q = QdrantClient(":memory:")
client_q.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

vector_store = QdrantVectorStore(
    client=client_q,
    collection_name="demo_collection",
    embedding=embeddings,
)
vector_store.add_documents(docs)

retriever = vector_store.as_retriever()

# --- LangSmith Prompt + LLM ---
langsmith_client = Client()
prompt = langsmith_client.pull_prompt("rlm/rag-prompt")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)


rag_chain = prompt | llm

# --- User Query ---
st.markdown("---")
question = st.chat_input("Your question:")

if question:
    with st.spinner("Generating answer..."):
        context = retriever.invoke(question)
        reply = rag_chain.invoke({"question": question, "context": context}).content
        st.success("Answer:")
        st.write(reply)

# --- Footer ---
st.markdown("---")
