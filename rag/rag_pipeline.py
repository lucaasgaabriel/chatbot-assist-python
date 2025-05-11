from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"}
        )
        self.vectorstore = None

    def load_documents(self, text: str):
        """Processa texto e cria vetores"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(text)
        self.vectorstore = FAISS.from_texts(chunks, self.embeddings)

    def query(self, question: str) -> str:
        """Busca resposta no RAG"""
        if not self.vectorstore:
            return "Erro: Documentos não carregados"
        
        docs = self.vectorstore.similarity_search(question, k=2)
        return "\n\n".join([d.page_content for d in docs])