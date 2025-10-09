"""Streamlit UI for interacting with the Profe Provisioning assistant.

This interface provides a minimal web UI to pose questions to the RAG pipeline.
It constructs the chain once and caches it in the Streamlit session state to
avoid re-initialising expensive components.
"""

from __future__ import annotations

import streamlit as st

from .chains import build_chain


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title="Profe Provisioning", layout="wide")
    st.title("Profe Provisioning — RAG Demo")

    # Build the chain once per session and store it in session state.
    if "chain" not in st.session_state:
        st.session_state.chain = build_chain(k=4)

    query = st.text_input("¿Qué buscás?")

    if st.button("Consultar") or query:
        q = query.strip()
        if not q:
            st.warning("Ingresá una consulta.")
        else:
            with st.spinner("Consultando..."):
                answer = st.session_state.chain.invoke(q)
            st.markdown("**Respuesta:**")
            st.write(answer)


if __name__ == "__main__":
    main()
