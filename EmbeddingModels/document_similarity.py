from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

documents = ["virat kholi is good player in crickey .","Ms.dhone is world besst wickety kiper and batsame also","sachine tholder are god to the cirkety"]
uer = "virat kholi"
doc_vector = model.embed_documents(
    documents
)

uer_vector = model.embed_query(uer)
sim = cosine_similarity([uer_vector],doc_vector)[0]
index , score = sorted(list(enumerate(sim)), key= lambda x : x[1])[-1]

print(uer)
print(documents[index])
print('similarity score is:',score)