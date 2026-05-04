from langchain_huggingface import HuggingFaceHub  # Optional HF inference
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.prompts import PromptTemplate
from knowledge_base import get_retriever

retriever = get_retriever()

# Use local prompt-based or HF model (no API key needed)
# For simplicity, use a rule-based + retrieval response with citations

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_agent_response(query):
    retriever = get_retriever()
    docs = retriever.invoke(query)
    context = format_docs(docs)

    prompt_template = """
    Kamu adalah ahli Pengelolaan BMN (Barang Milik Negara).
    Jawab pertanyaan berdasarkan konteks di bawah. Jika tidak tahu, katakan tidak ada info.
    Sertakan kutipan sumber.

    Konteks:
    {context}

    Pertanyaan: {question}

    Jawaban (dengan kutipan):
    """
    PROMPT = PromptTemplate.from_template(prompt_template)

    # Simplified for cloud

    # For now, use direct formatting as 'LLM' (extend with HF later)
    response = f"""
**Jawaban:**

{context[:1000]}...

**Kutipan Relevan:**
{chr(10).join([f'- {doc.metadata.get("source", "BMN Docs")}' for doc in docs[:3]])}"""
    
    return response

# Test: print(get_agent_response("Dokumen sewa BMN"))

