import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    api_key=google_api_key
)

pergunta = "Qual é a política de home office da nossa empresa?"
prompt_tradicional = ChatPromptTemplate.from_template(
    "Responda a seguinte pergunta: {pergunta}"
)

chain_tradicional = prompt_tradicional | llm
resposta_tradicional = chain_tradicional.invoke({"pergunta": pergunta})
print(resposta_tradicional.content)