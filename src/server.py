from fastapi import FastAPI
from pydantic import BaseModel
from chains import build_chain

app = FastAPI(title="Profe Provisioning — RAG API")
chain = build_chain()


class AskPayload(BaseModel):
    question: str


@app.post("/ask")
async def ask(payload: AskPayload):
    answer = await chain.ainvoke(payload.question)
    return {"answer": answer}
