import os
import duckdb
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredLoader

loader = UnstructuredLoader("./arquivos/relatorio_vendas.pdf", mode="elements")
docs_unstructured = loader.load()

docs_com_metadados = []

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

print(f"Total de elementos extraídos: {len(docs_unstructured)}\n")

for doc in docs_unstructured:
    print(f"--- TIPO DE ELEMENTO: {doc.metadata.get('category')} ---")
    print(doc.page_content)
    print("\n")


for doc in docs_unstructured:
    novos_metadados = doc.metadata.copy()
    novos_metadados['source'] = 'relatorio_vendas.pdf'
    novos_metadados['ingestion_date'] = datetime.now().strftime('%Y-%m-%d')
    novos_metadados['data_owner'] = 'Departamento de vendas'

    docs_com_metadados = Document(
        page_content=doc.page_content,
        metadata=novos_metadados
    )
    docs_com_metadados.append(docs_com_metadados)

    print(f"Total de documentos com metadados: {len(docs_com_metadados)}")
    print(docs_com_metadados[-1])
    

