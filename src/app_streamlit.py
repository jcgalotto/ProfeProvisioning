import streamlit as st
from chains import build_chain

st.set_page_config(page_title="Profe Provisioning", layout="wide")

st.title("📚 Profe Provisioning — Asistente de Documentación")

if "chain" not in st.session_state:
    st.session_state.chain = build_chain()

q = st.text_input("Preguntá algo sobre tus documentos")

if st.button("Consultar"):
    if not q.strip():
        st.warning("Escribí una pregunta.")
    else:
        with st.spinner("Pensando..."):
            answer = st.session_state.chain.invoke(q)
        st.markdown("### Respuesta")
        st.write(answer)
