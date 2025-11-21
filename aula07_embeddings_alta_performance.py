import os
import time
import numpy as np
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

textos_teste = [
    "Qual é a política de férias da nossa empresa?",
    "Preciso de um relatório de despesas de viagem",
    "Como configuro o acesso à rede privada virtual (VPN)?",
    "Onde encontro o código de conduta da organização?",
    "Quero entender o processo de avaliação de performance."
]

## Google Gemini Embeddings
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    #model="models/embedding-001"
    model="text-embedding-004"
)

start_time = time.time()
embeddings_gemini = gemini_embeddings.embed_documents(textos_teste)
end_time = time.time()
print(f"Tempo de processamento: {end_time - start_time} segundos")
print(f" - Dimensões do vetor: {len(embeddings_gemini[0])}")

## all-MiniLM-L6-v2 Embeddings
minilm_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

start_time = time.time()
embeddings_minilm = minilm_embeddings.embed_documents(textos_teste)
end_time = time.time()
print(f"Tempo de processamento: {end_time - start_time} segundos")
print(f" - Dimensões do vetor: {len(embeddings_minilm[0])}")

## BGE-Large Embeddings
bge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

start_time = time.time()
embeddings_bge = bge_embeddings.embed_documents(textos_teste)
end_time = time.time()
print(f"Tempo de processamento: {end_time - start_time} segundos")
print(f" - Dimensões do vetor: {len(embeddings_bge[0])}")