import sys
import os
import streamlit as st
import chromadb
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from config import *
import logging
import json
import asyncio



try:
    
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

sys.setrecursionlimit(3000)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AgroQuery AI", page_icon="🌾", layout="wide")

st.markdown("""
<style>
    /* Background */
    .stMainBlockContainer, .main, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f3ed 0%, #faf8f3 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #5A8F4C 0%, #4A7D3C 100%);
    }
    
    [data-testid="stSidebar"] button {
        background-color: #7DB757 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 11px 15px !important;
        margin: 6px 0 !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
    }
    
    [data-testid="stSidebar"] button:hover {
        background-color: #6BA447 !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Chat Title */
    .chat-title {
        font-size: 2.5rem;
        color: #5A8F4C;
        margin-bottom: 30px;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(90, 143, 76, 0.1);
    }
    
    /* Chat Container */
    .chat-container {
        background: white;
        border-radius: 18px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 6px 20px rgba(90, 143, 76, 0.12);
        border: 1px solid rgba(90, 143, 76, 0.08);
        max-height: 60vh;
        overflow-y: auto;
    }
    
    /* Messages */
    .message-group {
        display: flex;
        margin-bottom: 16px;
        gap: 10px;
    }
    
    .message-group.user {
        justify-content: flex-end;
    }
    
    .message-group.assistant {
        justify-content: flex-start;
    }
    
    .message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
    }
    
    .message-avatar.user {
        background: linear-gradient(135deg, #64b5f6 0%, #2196f3 100%);
    }
    
    .message-avatar.assistant {
        background: linear-gradient(135deg, #81c784 0%, #4caf50 100%);
    }
    
    .message-content {
        max-width: 70%;
        padding: 13px 18px;
        border-radius: 14px;
        word-wrap: break-word;
        line-height: 1.5;
        font-size: 14px;
    }
    
    .message-content.user {
        background: linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%);
        color: white;
        border-bottom-right-radius: 4px;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3);
    }
    
    .message-content.assistant {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        color: #2e7d32;
        border-bottom-left-radius: 4px;
        border: 1px solid #c8e6c9;
    }
    
    /* Input Area */
    input[type="text"] {
        background-color: white !important;
        color: #333 !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }
    
    input[type="text"]::placeholder {
        color: #888 !important;
        opacity: 1 !important;
    }
    
    input[type="text"]:focus {
        border-color: #5A8F4C !important;
        box-shadow: 0 0 0 4px rgba(90, 143, 76, 0.1) !important;
        outline: none !important;
    }
    /* Get Answer Button */
    .stButton > button {
        background: linear-gradient(135deg, #7DB757 0%, #5A8F4C 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(90, 143, 76, 0.3) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(90, 143, 76, 0.4) !important;
        background: linear-gradient(135deg, #6BA447 0%, #4A7D3C 100%) !important;
    }
    
    /* Source Citation */
    .source-citation {
        margin-top: 25px;
        padding: 15px 20px;
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        border-radius: 12px;
        font-size: 13px;
        color: #333;
        border-left: 4px solid #5A8F4C;
        box-shadow: 0 2px 8px rgba(90, 143, 76, 0.1);
        font-weight: 500;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] h2 {
        color: white !important;
    }
    
    /* Text visibility */
    h3 {
        color: #5A8F4C !important;
        font-weight: 700 !important;
    }
    
    /* Info box styling */
    [data-testid="stAlert"] {
        background-color: #e8f5e9 !important;
        color: #2e7d32 !important;
        border-radius: 12px !important;
        border-left: 4px solid #5A8F4C !important;
    }
    
    [data-testid="stAlert"] p {
        color: #2e7d32 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(ttl=3600)
def initialize_qa_system():
    try:
        logger.info("Initializing QA system...")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(client=chroma_client, collection_name=COLLECTION_NAME, embedding_function=embeddings)
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0)
        
        prompt_template ="""
You are an expert assistant for  India's agricultural and climate data, including:
- Crop production and area by state, district, year, season, and crop type;
- Rainfall and climate data by state, district, year, month, and agency.

You have access to tables of recent years of data from both domains.

========
Instructions:

1. When comparing regions or time periods, always provide summary tables (markdown or text), and include at least average values and totals.
2. When asked to list "top" crops or districts by production, order results by production volume and show both raw figures and rankings.
3. For trend or correlation questions, describe the production or rainfall trend (increase, decrease, volatility), and *if possible* explain statistically whether climate seems to impact agricultural patterns, citing data years and quantifying relationships.
4. For policy advice, give specific data-backed arguments—reference historic crop yields, rainfall patterns, and their implications, including both positive and negative examples where relevant.
5. Always clearly cite your sources as (Indian Government agri & climate data), noting tables or figure extracts when possible.

========
Context:
{context}

========
Question:
{question}

========
Your answer:
"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3,"fetch_k": 6,"lambda_mult": 0.7}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        logger.info("QA system initialized successfully!")
        return qa_chain, vectorstore
    except Exception as e:
        logger.error(f"Error initializing QA system: {e}")
        st.error(f"Failed to initialize system: {e}")
        return None, None

@st.cache_resource(ttl=1800)
def load_documents():
    try:
        with open("data/processed/documents.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
        return docs
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        st.error("Error loading mandi data.")
        return None

def format_sources(source_documents):
    # Always cite Agmarknet/Government of India
    return ["Agmarknet (Government of India) market arrivals and prices"]

def main():
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown('<div style="text-align: center; margin-bottom: 24px; padding: 16px; background-color: rgba(255,255,255,0.1); border-radius: 8px;">'
                   '<span style="font-size: 24px;"></span> <span style="color: white; font-weight: 700; font-size: 18px;">🌾AgroQuery AI</span>'
                   '<p style="color: rgba(255,255,255,0.9); font-size: 12px; margin-top: 4px;">India Agri-Tech Intelligence</p>'
                   '</div>', unsafe_allow_html=True)
        
        st.markdown('<hr style="border-color: rgba(255,255,255,0.2);">', unsafe_allow_html=True)
      
        # About Section
        st.markdown('<div class="sidebar-title">About</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="color: rgba(255,255,255,0.95); font-size: 13px; line-height: 1.6;">
        Your AI-powered intelligence system for agriculture, rainfall, and crop insights across India.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<hr style="border-color: rgba(255,255,255,0.2); margin: 20px 0;">', unsafe_allow_html=True)
        
        # Sample Questions
        st.markdown('<div class="sidebar-title">Sample Questions</div>', unsafe_allow_html=True)
        sample_questions = [
            "Wheat production by state?",
            "Rainfall patterns this season?",
            "Best crops for monsoon?",
            "Crop yield trends 2024?"
        ]
        
        cols = st.columns(2)
        for idx, q in enumerate(sample_questions):
            with cols[idx % 2]:
                if st.button(q, key=f"sample_{idx}", use_container_width=True):
                    st.session_state['current_question'] = q
                    st.rerun()
        
        st.markdown('<hr style="border-color: rgba(255,255,255,0.2); margin: 20px 0;">', unsafe_allow_html=True)
    
    # Main Chat Area
    st.markdown('<div class="chat-title">🌾AgroQuery AI Chat</div>', unsafe_allow_html=True)
    
    # Chat Container
    chat_container = st.container()
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="message-group user">
                    <div class="message-content user">{message['content']}</div>
                    <div class="message-avatar user">👤</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-group assistant">
                    <div class="message-avatar assistant">🤖</div>
                    <div class="message-content assistant">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Input Area
    st.markdown("<div style='margin-top: 60px;'></div>", unsafe_allow_html=True)
    
    user_input = st.text_input(
        "Ask about mandi prices, arrivals, markets...",
        value=st.session_state.get('current_question', ""),
        key="user_input",
        placeholder="Example: What are the top crops in India by production?"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Get Answer", use_container_width=True, key="get_answer"):
            if user_input:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                with st.spinner("🔍 Fetching answer..."):
                    qa_chain, vectorstore = initialize_qa_system()
                    if qa_chain is None:
                        st.error("QA system initialization error. Check logs.")
                        return
                    
                    result = qa_chain({"query": user_input})
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": result["result"]})
                
                st.session_state.current_question = ""
                st.rerun()

    
    with col2:
        # Clear Chat Button
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.current_question = ""
            st.rerun()
        
    st.markdown("### Waiting for your question")
    st.info("Enter a question and click 'Get Answer' — results will show here.")
    # Source Citation
    if st.session_state.chat_history:
        st.markdown("<div class='source-citation'>"
                   "📌 <strong>Source:</strong> Indian Government Agriculture & Climate Data"
                   "</div>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()