import streamlit as st
from knowledge_base import get_retriever
from rag_agent import get_agent_response

st.set_page_config(page_title="BMN AI Agent", page_icon="🏛️", layout="wide")

st.title("🏛️ AI Agent Pengelolaan BMN")
st.markdown("Tanyakan tentang siklus BMN, dokumen sewa, penetapan status, pemindahtanganan, dll.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Masukkan pertanyaan Anda mengenai BMN..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("AI sedang memproses..."):
            response = get_agent_response(prompt)
        st.markdown(response)

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.markdown("### Instruksi")
st.sidebar.info("1. Install deps: `pip install -r requirements.txt`\n2. Jalankan: `streamlit run app.py`\n3. Tanya e.g., 'Dokumen sewa BMN apa saja?'")

