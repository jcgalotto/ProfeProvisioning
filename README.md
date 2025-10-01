# Profe Provisioning — RAG Doc Assistant

## 1) Preparar entorno
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Editá `.env` con tu modelo y provider.

## 2) Agregar documentos

Poné PDFs en `data/pdfs/` y notas `.md`/`.txt` en `data/md/`.

## 3) Ingesta

```bash
python -m src.ingest
```

## 4) Probar API

```bash
uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
# POST a /ask con {"question": "..."}
```

## 5) UI Streamlit

```bash
streamlit run src/app_streamlit.py
```

## 6) Tips

* Ajustá `CHUNK_SIZE`/`CHUNK_OVERLAP` según el tipo de documento.
* Para respuestas más exactas, bajá `temperature` del LLM.
* Para datasets grandes, preferí Chroma con `persist_directory`.
* Podés agregar citations devolviendo metadatos de cada chunk.
