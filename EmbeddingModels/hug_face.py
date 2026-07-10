from langchain_huggingface import HuggingFaceEmbeddings

# embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# text = "delhi is the capital of india"

# vector = embedding.embed_query(text)

# print(str(vector))

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

doc = ["delhi is the capital of india","what is deep learning","who i can learn ai"]

vector = embedding.embed_documents(doc)

print(str(vector))