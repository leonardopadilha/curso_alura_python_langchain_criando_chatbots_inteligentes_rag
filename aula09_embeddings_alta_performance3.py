import os
import time
import numpy as np
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.storage import LocalFileStore
from langchain_community.embeddings import CacheBackedEmbeddings

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

textos_teste = [
    "Qual é a política de férias da nossa empresa?",
    "Preciso de um relatório de despesas de viagem",
    "Como configuro o acesso à rede privada virtual (VPN)?",
    "Onde encontro o código de conduta da organização?",
    "Quero entender o processo de avaliação de performance."
]

store = LocalFileStore("./cache/")

embedder_principal = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

cache_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embedder_principal,
    store,
    namespace="gemini_cache"
)

textos_para_cache = ["Olá, mundo!", "Testando o cache de embeddings.", "Olá, mundo!"]
start_time = time.time()
embeddings_result_1 = cache_embeddings.embed_documents(textos_para_cache)
end_time = time.time()
print(f" Tempo de execução: {end_time - start_time:.4f} segundos.")

documentos_grandes = [f"Este é o documento de teste número {i}." for i in range(1000)]
bge_embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

batch_sizes = [1, 32, 64, 128]
for batch_size in batch_sizes:
    start_time = time.time()
    num_batches = len(documentos_grandes) // batch_size
    tempo_estimado = num_batches * (0.1 * batch_size) + (len(documentos_grandes) % batch_size) * 0.1
    tempo_real = bge_embedder.client.encode(documentos_grandes, batch_size=batch_size)
    end_time = time.time()
    print(f" - Batch Size: {batch_size:<4} -> Tempo: {end_time - start_time:.2f}s")
