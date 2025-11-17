import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
#from langchain_community.vectorstores import Chroma
#import faiss

# NOTA: ChromaDB está tendo problemas de travamento no Windows
# Usando FAISS como alternativa mais estável
print("⚠ NOTA: Usando FAISS em vez de ChromaDB devido a problemas de travamento no Windows")
print("   Para usar ChromaDB, descomente as linhas relacionadas e comente o código FAISS\n")

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

print("Inicializando embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    #model="models/embedding-001"
    model="text-embedding-004"
)
print("Embeddings inicializados!")

# Testando se os embeddings funcionam antes de criar o Chroma
print("Testando criação de embedding...")
try:
    teste_embedding = embeddings.embed_query("Teste")
    print(f"Embedding de teste criado com sucesso! Dimensão: {len(teste_embedding)}")
except Exception as e:
    print(f"ERRO ao criar embedding de teste: {e}")
    raise

pergunta = "Como solicito minhas férias?"

print("\n" + "="*60)
print("CRIANDO BANCO DE DADOS VETORIAL COM FAISS")
print("="*60)
print(f"Total de documentos: {len(documentos_empresa)}")

# Criando banco FAISS (mais estável no Windows)
print("\nCriando banco FAISS...")
print("NOTA: Isso pode demorar alguns segundos enquanto cria os embeddings...")

try:
    faiss_db = FAISS.from_documents(documentos_empresa, embeddings)
    print("✓ Banco de dados FAISS criado com sucesso!")
except Exception as e:
    print(f"✗ Erro ao criar banco FAISS: {e}")
    import traceback
    traceback.print_exc()
    raise

print(f"\nPergunta: '{pergunta}'")
print("Buscando documentos similares...")
resultados = faiss_db.similarity_search(pergunta, k=2)
print(f"Encontrados {len(resultados)} documentos relevantes:\n")
for doc in resultados:
    print(f"- {doc.page_content}")
    print(f"  (Metadados: {doc.metadata})")

# ============================================================================
# CÓDIGO DO CHROMA COMENTADO (devido a problemas de travamento no Windows)
# ============================================================================
"""
# SOLUÇÃO ALTERNATIVA COM CHROMA (se quiser tentar novamente):
# 
# print("\nCriando banco de dados Chroma...")
# print(f"Total de documentos: {len(documentos_empresa)}")
# 
# # Criar embeddings manualmente primeiro
# print("Passo 1: Criando embeddings para os documentos...")
# textos = [doc.page_content for doc in documentos_empresa]
# vetores_embeddings = embeddings.embed_documents(textos)
# print(f"✓ Embeddings criados! Total: {len(vetores_embeddings)}")
# 
# # Tentar criar Chroma
# print("\nPasso 2: Criando banco Chroma...")
# chroma_db = Chroma.from_texts(
#     texts=textos,
#     embedding=embeddings,
#     metadatas=[doc.metadata for doc in documentos_empresa],
#     collection_name="documentos_empresa"
# )
# 
# resultados = chroma_db.similarity_search(pergunta, k=2)
"""
