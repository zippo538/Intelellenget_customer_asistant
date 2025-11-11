import streamlit as st 
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from qdrant_client import QdrantClient
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
    print(f"Connect Ke Qdrant {url}")
    retriever = qdrant.as_retriever(search_kwargs={"k" : k})
    print(f"success retriever qdrant {collection_name}")
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
retriever = get_qdrant_retriever(collection_name="user_manual_BNI_Mbank")

def has_context(inputs) : 
    context = inputs.get("context","")
    return bool(context and context.strip())


context_q_system_prompt = "Kamu adalah asisten AI yang membantu pengguna menjawab pertanyaan tentang BNI Mobile Banking. \
    Gunakan konteks dari dokumen (hasil pencarian retriever) dan riwayat percakapan sebelumnya \
    untuk menjawab dengan akurat, ringkas, dan relevan. \
    Jika konteks tidak mengandung informasi yang cukup untuk menjawab, katakan dengan sopan 'Maaf, saya tidak tahu.'"

context_q_user_prompt = "Pertanyaan: {input}\n\n Konteks:\n{context}"

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


qa_chain = create_stuff_documents_chain(
    model_llm,
    context_q_prompt,
)

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain =qa_chain
)


no_context_chain = RunnableLambda(lambda x : {"answer" : "Maaf saya tidak tahu"})

final_chain = RunnableBranch(
    (has_context,rag_chain),
    RunnablePassthrough() | no_context_chain
    | StrOutputParser()
)



st.set_page_config(page_title="Chat Bot BNI MOBILE BANKING",page_icon="🤖", layout="centered")

st.sidebar.title("BNI MOBILE BANKING CHAT BOT🤖")
st.sidebar.markdown("### 🧠 Chat Memory")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("💬 AI ASSISTANT BNI MOBILE ")
st.caption("Ask anything — your AI assistant is here to help!")




def format_chat_history(chat_history):
    """
    Ubah list of AIMessage/HumanMessage menjadi string percakapan.
    """
    history_str = ""
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            history_str += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_str += f"AI: {msg.content}\n"
    return history_str.strip()


#tampilkan riwayat percakapan
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        role = "user"
        content_str = f"**User:** {msg.content}"
    elif isinstance(msg, AIMessage):
        role = "assistant"
        content_str = f"**AI:** {msg.content}"
    else:
        role = "system" 
        content_str = f"**System:** {msg.content}" if hasattr(msg, 'content') else str(msg)
         
    with st.chat_message(role):
        st.markdown(msg.content)


#input pengguna
if prompt := st.chat_input("Ketik pesan disini..."):
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    docs = retriever.invoke(prompt)
    context = "\n\n".join([d.page_content for d in docs]) if docs else "Saya tidak tahu"


    #response manusia
    with st.chat_message("user"):
        st.markdown(prompt)
    
    #response ai
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("_AI sedang mengetik..._")
        
        response = final_chain.invoke({
            "input":prompt,
            "chat_history" :st.session_state.chat_history,
            "context" : context
            })
        
        message_placeholder.markdown(response['answer'])
        print(response)
    
    
    st.session_state.chat_history.append(AIMessage(content=response['answer']))
    
    
    



