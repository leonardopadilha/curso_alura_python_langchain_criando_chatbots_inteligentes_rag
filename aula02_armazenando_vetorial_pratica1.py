import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    #model="models/embedding-001"
    model="text-embedding-004"
)

resposta =embeddings.embed_query("Política de home office da empresa?")
print(resposta)