from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

result = model.invoke("Search the Name of Amit Kumar Singh Kurmi  study at Kalinga University form nepal , if you find then  yes if not then sya NO,")

print(result.content)