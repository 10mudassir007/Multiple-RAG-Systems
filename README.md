# Multiple-RAG-Systems

🚀 **Multiple-RAG-Systems** is a comprehensive collection of **Retrieval-Augmented Generation (RAG)** implementations, designed for various levels of complexity. It features **basic RAGs, advanced RAGs, and a simple RAG implementation using LlamaIndex**. This repository serves as a go-to resource for developers exploring different RAG techniques using **LangChain, LangGraph, Firecrawl, Crawl4AI, Qdrant, Groq, Mistral, Tavily Search, and more**.

## 📌 Folder Structure

### 1️⃣ Basic RAGs

The **Basic RAGs** folder contains foundational RAG implementations tailored for different use cases:

#### 📂 `DocLens`
- **`rag_streamlit.py`** – A document-based RAG system with a Streamlit UI for easy interaction.

#### 📂 `FindWise`
- **`search_rag.py`** – A search-based RAG that retrieves information from the internet and generates responses.

#### 📂 `WebFusion`
- **`rag_webbased.py`** – A URL-based RAG that processes and generates responses from web content.

---

### 2️⃣ Advanced RAGs

The **Advanced RAGs** folder contains sophisticated implementations in Jupyter notebooks:

#### 📜 `agentic_rag.ipynb`
- Implements an **Agentic RAG** using **LangChain and LangGraph** with tool integration.

#### 📜 `agentic_rag_langgraph.ipynb`
- Advanced **Agentic RAG** with **reranking, multiquery retriever, and Tavily Search** using LangChain and LangGraph.

#### 📜 `corrective_rag.ipynb`
- Checks **document relevance** and, if insufficient, **modifies the query** and performs a **web search** for better results.

#### 📜 `fusion_rag_firecrawl_crawl4ai.ipynb`
- Implements **multi-retriever fusion RAG** using **Firecrawl and Crawl4AI**, stores results in **Qdrant**, and generates responses using **Mistral**.

#### 📜 `github_Loader_and_rag_evaluation.ipynb`
- Demonstrates a **GitHub Loader** in LangChain and evaluates RAG outputs using **Opik, Ragas, and Giskard**.

#### 📜 `rag_reranking_techniques.ipynb`
- Showcases multiple **reranking techniques** for improving RAG responses.

#### 📜 `speculative_rag.ipynb`
- Generates multiple responses using **multi-query techniques** and selects the best one through scoring.

---

### 3️⃣ RAG with LlamaIndex

The **With LlamaIndex** folder contains a simple RAG implementation:

#### 📜 `naive_rag_llama_index.ipynb`
- A **lightweight RAG setup** using **LlamaIndex** for retrieval and response generation.

---

## 🚀 Getting Started

### 1️⃣ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/rag-library.git
cd rag-library

## 📜 License

This project is licensed under the **MIT License**.

---

🚀 **RAG-Library** – Powering intelligent and scalable Retrieval-Augmented Generation workflows!
