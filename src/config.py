import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

    EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai")
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")

    VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma")
    PERSIST_DIR = os.getenv("PERSIST_DIR", "storage")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))


settings = Settings()
