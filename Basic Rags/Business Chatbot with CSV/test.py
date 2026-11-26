from langchain_huggingface.embeddings import HuggingFaceEmbeddings
emmbeddings = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-small-english-r2")











query = "List my top 10 clients"
a = emmbeddings.embed_query(query)
print(a)