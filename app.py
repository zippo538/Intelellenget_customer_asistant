import streamlit as st 
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptTemplate
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv("GROQ_API_KEY")


def local_css(file_name):
    with open(file_name) as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>",unsafe_allow_html=True)

def get_qdrant_retriever(url: str = "http://localhost:6333", collection_name: str = None, model_name : str = "all-MiniLM-L6-v2", k : int = 4):
       
    embedding = SentenceTransformerEmbeddings(model_name=model_name)
    client = QdrantClient(url=url)
    
    qdrant = Qdrant(
        client=client,
        collection_name=collection_name, # <== Penentuan COLLECTION
        embeddings=embedding,
    )
    retriever = qdrant.as_retriever(search_kwargs={"k" : k})
    return retriever



def load_llm(id_model, temperature):
    llm = ChatGroq(
        model=id_model,
        temperature=temperature,
        api_key=api_key,
        max_tokens=1024,
        timeout=None,
        max_retries=3
    )
    return llm

model_llm = load_llm('meta-llama/llama-4-maverick-17b-128e-instruct', 0.3)
retriever = get_qdrant_retriever("user_manual_BNI_Mbank")



context_q_system_prompt = "Given the following chat history and the follow-up question which might reference \
    context in the chat history, rephrase the follow-up question to be a standalone question which can be unserstood without the chat history. \
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    
context_q_system_prompt = "Given the following chat history and the follow-up question which might reference \
    context in the chat history, rephrase the follow-up question to be a standalone question which can be unserstood without the chat history. \
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is."

context_q_user_prompt = "Question : {input}"

context_q_prompt = ChatPromptTemplate.from_messages([
    ("system",context_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human",context_q_user_prompt)
])

history_aware_retriever = create_history_aware_retriever(
    llm=model_llm,
    retriever=retriever,
    prompt=context_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",context_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "Question: {input}\nContext: {context}\nPlease provide a concise and accurate answer based on the context provided. \
        If the context does not contain sufficient information to answer the question, respond with \"I don't know.\""), 
])

qa_chain = create_stuff_documents_chain(
    model_llm,
    qa_prompt,
)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    qa_chain
)



st.set_page_config(page_title="Chat Bot BNI MOBILE BANKING",page_icon="🤖", layout="wide")

st.sidebar.title("BNI MOBILE BANKING CHAT BOT🤖")
st.sidebar.markdown("### 🧠 Chat Memory")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("💬 AI Assistant")
st.caption("Ask anything — your AI assistant is here to help!")

if st.session_state.chat_history:
    chat_text = "\n\n".join(
        [f"HumanMessage: {msg['content']}" if msg["role"] == "HumanMessage" else f"AIMessage: {msg['content']}" for msg in st.session_state.chat_history]
    )

for msg in st.session_state.chat_history:
    if msg["role"] == "HumanMessage" : 
        st.markdown(f"<div class='stChatMessage user'>🧑‍💻: {msg['content']}</div>", unsafe_allow_html=True)
    else : 
        st.markdown(f"<div class='stChatMessage bot'>🤖: {msg['content']}</div>", unsafe_allow_html=True)

with st.form("chat_form",clear_on_submit=True):
    user_input = st.text_input("Type your message", key="input",placeholder="Adakah bisa saya Bantu?")
    submitted = st.form_submit_button("send")

if submitted and user_input : 
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    result = rag_chain.invoke({"input":user_input,"chat_history":st.session_state.chat_history})
    st.session_state.chat_history.append(AIMessage(content=result["answer"]))
    
    st.rerun()
    
    



