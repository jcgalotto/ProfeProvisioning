from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from retriever import build_retriever
from config import settings

SYSTEM_PROMPT = """
Eres un asistente técnico. Usa EXCLUSIVAMENTE la información de los documentos recuperados.
Si no está en las fuentes, admite que no lo sabes.
Responde en español, claro y conciso, con ejemplos cuando aplique.
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Pregunta: {question}\n\nContexto:\n{context}\n\nRespuesta concisa:")
])


def format_docs(docs):
    return "\n\n".join([f"[Chunk {i+1}]\n" + d.page_content for i, d in enumerate(docs)])


def _build_llm():
    if settings.LLM_PROVIDER.lower() == "openai":
        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY no configurada y LLM_PROVIDER=openai. "
                "Definí la key en .env o cambiá LLM_PROVIDER a 'hf'."
            )
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0
        )
    else:
        # Placeholder para futuros providers (HF/local)
        raise NotImplementedError("LLM_PROVIDER distinto de 'openai' aún no implementado en chains.py")


def build_chain():
    retriever = build_retriever(k=4)
    llm = _build_llm()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
