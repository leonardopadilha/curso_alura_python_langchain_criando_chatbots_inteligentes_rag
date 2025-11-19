import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec

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

google_api_key = os.getenv("PINECONE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    #model="models/embedding-001"
    model="text-embedding-004"
)

d = 768
pergunta = "Quais são as regras da empresa?"

pinecone_client = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
spec = ServerlessSpec(cloud="aws", region="us-east-1")

# Verificando se o índice existe. Se não, ele será criado.
# Nome do seu índice no Pinecone
index_name = "langchain-rag"
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=index_name,
        dimension=d,
        metric="cosine",
        spec=spec
    )
    print(f"Índice {index_name} criado com sucesso no Pinecone.")

    pinecone_db = Pinecone.from_documents(
        documentos_empresa,
        embeddings,
        index_name=index_name
    )

    print(f"Documentos inseridos no índice {index_name} no Pinecone.")

else:
    print(f"Conectando ao índice existente '{index_name}'.")
    pinecone_db = Pinecone.from_existing_index(
        index_name,
        embeddings
    )


if pinecone_db:
    # Busca por similaridade
    pergunta_ti = "Como configuro a VPN?"
    resultados_pinecone = pinecone_db.similarity_search(pergunta_ti, k=2)
    print(f"\nPergunta: '{pergunta_ti}'")
    print("\nDocumentos mais relevantes (Pinecone):")
    for doc in resultados_pinecone:
        print(f"- {doc.page_content}")
        print(f" (Metadados: {doc.metadata})")

    # Busca com filtro (Pinecone também suporta)
    resultados_pinecone_filtrados = pinecone_db.similarity_search(
        "informações sobre regras",
        k=2,
        filter={"tipo": "política"}
    )

    print(f"\n Pergunta: 'Informações sobre regras' com filtro para tipo='política'")
    print(f"\n Documentos relevantes e filtrados (Pinecone):")
    for doc in resultados_pinecone_filtrados:
        print(f"- {doc.page_content}")
        print(f"(Tipo: {doc.metadata['tipo']})")


