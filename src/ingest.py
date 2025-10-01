import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None
from langchain_community.vectorstores import Chroma, FAISS
from config import settings

DATA_DIRS = ["data/pdfs", "data/md"]


def _load_documents():
    docs = []
    for d in DATA_DIRS:
        p = Path(d)
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.suffix.lower() in [".pdf"]:
                loader = PyPDFLoader(str(f))
                docs.extend(loader.load())
            elif f.suffix.lower() in [".md", ".txt"]:
                loader = TextLoader(str(f), encoding="utf-8")
                docs.extend(loader.load())
    return docs


def _base_embeddings():
    provider = settings.EMBEDDINGS_PROVIDER.lower()
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY no configurada y EMBEDDINGS_PROVIDER=openai. "
                "Definí la key en .env o cambiá EMBEDDINGS_PROVIDER a 'hf'."
            )
        return OpenAIEmbeddings(model=settings.EMBEDDINGS_MODEL, api_key=settings.OPENAI_API_KEY)
    elif provider == "hf":
        if HuggingFaceEmbeddings is None:
            raise RuntimeError("Instalá 'sentence-transformers' para usar HuggingFaceEmbeddings")
        return HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL)
    else:
        raise ValueError(f"Provider de embeddings no soportado: {provider}")


def _get_embeddings():
    base = _base_embeddings()
    store = InMemoryByteStore()
    return CacheBackedEmbeddings.from_bytes_store(base, store)


def run_ingest():
    print("Cargando documentos...")
    documents = _load_documents()
    if not documents:
        raise SystemExit("No hay documentos en data/. Subí PDFs o MD/TXT.")

    print(f"Docs: {len(documents)}. Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Chunks: {len(chunks)}")

    print("Embeddings...")
    embeddings = _get_embeddings()

    print(f"Creando vector store: {settings.VECTOR_STORE}")
    if settings.VECTOR_STORE == "chroma":
        vs = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory=settings.PERSIST_DIR,
            collection_name="docs"
        )
        vs.persist()
    else:
        vs = FAISS.from_documents(chunks, embedding=embeddings)
        faiss_path = os.path.join(settings.PERSIST_DIR, "faiss_index")
        Path(settings.PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        vs.save_local(faiss_path)

    print("Listo.")


if __name__ == "__main__":
    run_ingest()
