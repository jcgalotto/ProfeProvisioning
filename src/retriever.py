from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None
from config import settings


def _embeddings():
    if settings.EMBEDDINGS_PROVIDER == "openai":
        return OpenAIEmbeddings(model=settings.EMBEDDINGS_MODEL)
    else:
        if HuggingFaceEmbeddings is None:
            raise RuntimeError("Instala sentence-transformers para usar HF embeddings")
        return HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL)


def build_retriever(k: int = 4):
    if settings.VECTOR_STORE == "chroma":
        vs = Chroma(
            persist_directory=settings.PERSIST_DIR,
            collection_name="docs",
            embedding_function=_embeddings()
        )
    else:
        vs = FAISS.load_local(
            f"{settings.PERSIST_DIR}/faiss_index",
            _embeddings(),
            allow_dangerous_deserialization=True
        )
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": max(k * 2, 8)})
    return retriever
