from fastapi import FastAPI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os

app = FastAPI()

# Configurações
DOCS_PATH = "/rag/data/python-3.13-docs-text"
VECTORSTORE_PATH = "/rag/data/vectorstore"

# Inicializa o RAG
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def load_documents():
    """Carrega e processa os documentos Python"""
    docs = []
    for root, _, files in os.walk(DOCS_PATH):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    docs.append(f.read())
    return text_splitter.create_documents(docs)

# Verifica se o vectorstore já existe
if os.path.exists(VECTORSTORE_PATH):
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings)
else:
    documents = load_documents()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

# Conexão com Ollama local
llm = Ollama(
    # base_url="http://host.docker.internal:11434",  # Mac/Windows
    base_url="http://172.17.0.1:11434",  # Linux
    model="gemma3",
    temperature=0.3
)

# Template otimizado para Gemma 3 e documentação Python
RAG_PROMPT_TEMPLATE = """[INST] <<SYS>>
Você é um assistente especializado em documentação Python 3.13.
Responda de forma técnica e precisa usando APENAS o contexto fornecido.
<</SYS>>

Contexto relevante:
{context}

Pergunta: {question} 

Dê uma resposta completa com exemplos de código quando aplicável. [/INST]"""

prompt = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
    verbose=True
)

@app.post("/ask")
async def ask_question(question: str):
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.metadata.get("source", "") for doc in result["source_documents"]]
    }