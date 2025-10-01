import os
from src.ingest import run_ingest
from src.chains import build_chain


def test_pipeline():
    if not os.path.exists("storage"):
        os.makedirs("storage", exist_ok=True)
    run_ingest()
    chain = build_chain()
    out = chain.invoke("¿De qué tratan los documentos?")
    assert isinstance(out, str) and len(out) > 0
