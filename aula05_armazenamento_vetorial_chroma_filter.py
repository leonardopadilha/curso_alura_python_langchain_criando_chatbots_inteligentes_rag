import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
#from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
#import faiss

documentos_empresa = [
    Document(
        page_content="Política de férias: Funcionários têm direito a 30 dias de férias após 12 meses. A solicitação deve ser feita com 30 dias de antecedência.",
        metadata={"tipo": "política", "departamento": "RH", "ano": 2024, "id_doc": "doc001"}
    ),
    Document(
        page_content="Processo de reembolso de despesas: Envie a nota fiscal pelo portal financeiro. O reembolso ocorre em até 5 dias úteis.",
        metadata={"tipo": "processo", "departamento": "Financeiro", "ano": 2023, "id_doc": "doc002"}
    ),
    Document(
        page_content="Guia de TI: Para configurar a VPN, acesse vpn.nossaempresa.com e siga as instruções para seu sistema operacional.",
        metadata={"tipo": "tutorial", "departamento": "TI", "ano": 2024, "id_doc": "doc003"}
    ),
    Document(
        page_content="Código de Ética e Conduta: Valorizamos o respeito, a integridade e a colaboração. Casos de assédio não serão tolerados.",
        metadata={"tipo": "política", "departamento": "RH", "ano": 2022, "id_doc": "doc004"}
    )
]

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    #model="models/embedding-001"
    model="text-embedding-004"
)

#resposta =embeddings.embed_query("Política de home office da empresa?")
#print(resposta)

d = 768
#index_hnsw = faiss.IndexHNSWFlat(d, 32)
#faiss_db = FAISS.from_documents(documentos_empresa, embeddings)

pergunta = "Quais são as regras da empresa?"

#resultados = faiss_db.similarity_search(pergunta, k=2)
#print(f"\nPergunta: '{pergunta}'")
#print("\nDocumentos mais relevantes (FAISS):")
#for doc in resultados:
#    print(f"- {doc.page_content}")
#    print(f" (Metadados: {doc.metadata})")

chroma_db = Chroma.from_documents(
    documents=documentos_empresa,
    embedding=embeddings,
    # persist_directory="chroma_db"
)

resultados = chroma_db.similarity_search(
    pergunta, 
    k=2,
    filter={"$and": [{"departamento": "RH"}, {"tipo": "política"}]}    
)

for doc in resultados:
    print(f"- {doc.page_content}")
    #print(f" (Metadados: {doc.metadata})")
