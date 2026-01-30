import streamlit as st
import os
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Mahabharat GPT",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. THEME & DESIGN SYSTEM (CSS) ---
st.markdown("""
    <style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Karma:wght@300;400;600&display=swap');

    /* COLOR PALETTE */
    :root {
        --saffron-deep: #D35400;
        --maroon-royal: #641E16;
        --gold-antique: #D4AF37;
        --parchment-bg: #F5E6CC;
        --parchment-light: #FAF3E0;
        --text-dark: #2C1909;
        --deep-wood: #2c1505;
    }

    /* GLOBAL APP STYLING */
    .stApp {
        background-color: var(--parchment-bg);
        background-image: radial-gradient(#d4af37 1px, transparent 1px);
        background-size: 30px 30px;
        font-family: 'Karma', sans-serif;
    }

    /* --- SIDEBAR STYLING (The Wood Look) --- */
    section[data-testid="stSidebar"] {
        background-color: var(--deep-wood);
        border-right: 3px solid var(--gold-antique);
    }
    
    /* Force Sidebar Text Color to Cream (So it's visible) */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: var(--parchment-light) !important;
    }

    /* FIX: Sidebar Buttons (Prevent Text Overflow) */
    .stButton > button {
        width: 100%;
        background-color: var(--maroon-royal);
        color: var(--gold-antique) !important;
        border: 1px solid var(--gold-antique);
        font-family: 'Cinzel', serif;
        /* Crucial for Mobile: Allows text to wrap if it's too long */
        white-space: normal !important; 
        height: auto !important;
        padding: 0.5rem 1rem !important;
        margin-bottom: 5px;
    }
    
    .stButton > button:hover {
        background-color: var(--saffron-deep);
        border-color: #FFF;
        color: #FFF !important;
    }

    /* --- CHAT INTERFACE --- */
    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem;
        border-bottom: 2px solid var(--gold-antique);
        margin-bottom: 2rem;
        background: linear-gradient(to right, transparent, rgba(212, 175, 55, 0.1), transparent);
    }
    .main-header h1 {
        font-family: 'Cinzel', serif;
        color: var(--maroon-royal);
        text-transform: uppercase;
        font-size: 2.5rem;
        margin: 0;
    }
    
    /* Text Visibility in Main Area */
    .stMarkdown, p, span, div {
        color: var(--text-dark);
    }

    /* Chat Bubbles */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: rgba(255, 255, 255, 0.6);
        border: 1px solid var(--saffron-deep);
        border-radius: 15px 15px 0px 15px;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #E8EAF6; 
        border-left: 4px solid var(--gold-antique);
        border-radius: 0px 15px 15px 15px;
    }

    /* --- MOBILE RESPONSIVENESS --- */
    @media only screen and (max-width: 600px) {
        .main-header h1 {
            font-size: 1.5rem !important; /* Smaller title on phone */
        }
        .stChatMessage {
            font-size: 0.9rem; /* Readable text on small screens */
        }
        /* Adjust sidebar width behavior for mobile is handled by Streamlit, 
           but we ensure padding is safe */
        .block-container {
            padding-top: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOGIC SETUP ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Missing API Key. Please add it to Streamlit Secrets.")
    st.stop()

DB_PATH = "./mahabharat_db"

@st.cache_resource
def get_engine():
    if not os.path.exists(DB_PATH):
        return None, None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.4,
        convert_system_message_to_human=True
    )
    return retriever, llm

retriever, llm = get_engine()

# --- 4. SIDEBAR NAVIGATION (CLEANER & MOBILE READY) ---
with st.sidebar:
    st.markdown("## 📜 Chronicles")
    st.markdown("Select a path:")
    
    st.markdown("---")
    
    # Using Expanders to organize the list nicely
    with st.expander("⚜️ The Pandavas", expanded=True):
        if st.button("Yudhishthira (Dharma)"):
            st.session_state.prompt_input = "Tell me about Yudhishthira's adherence to Truth and the game of dice."
        if st.button("Bhima (Strength)"):
            st.session_state.prompt_input = "Describe Bhima's immense strength and his vows."
        if st.button("Arjuna (Warrior)"):
            st.session_state.prompt_input = "Describe Arjuna's skills, the Gandiva bow, and his bond with Krishna."
        if st.button("Nakula & Sahadeva"):
            st.session_state.prompt_input = "What were the special skills and roles of Nakula and Sahadeva?"
        if st.button("Draupadi (Panchali)"):
            st.session_state.prompt_input = "Tell me about Draupadi's birth from fire and her resilience."

    with st.expander("🐍 The Kauravas"):
        if st.button("Duryodhana (King)"):
            st.session_state.prompt_input = "Explain Duryodhana's motivations, his jealousy, and his friendship with Karna."
        if st.button("Karna (Tragic Hero)"):
            st.session_state.prompt_input = "Tell me about Karna's tragic life, his charity (Daan), and his armor (Kavach)."
        if st.button("Shakuni (Planner)"):
            st.session_state.prompt_input = "What was Shakuni's role in the game of dice and influencing Duryodhana?"
        if st.button("Dushasana"):
            st.session_state.prompt_input = "What was Dushasana's role in the court and his fate in the war?"

    with st.expander("👴 Elders & Divine"):
        if st.button("Krishna (Divine)"):
            st.session_state.prompt_input = "Describe the role of Krishna as the charioteer and his divinity."
        if st.button("Bhishma (Grandsire)"):
            st.session_state.prompt_input = "What was Bhishma's vow of celibacy and his role as the commander?"
        if st.button("Guru Drona"):
            st.session_state.prompt_input = "Tell me about Dronacharya as the teacher of both clans and his death."

    st.markdown("---")
    st.info("💡 **Tip:** Type in English or Hinglish!")

# --- 5. MAIN INTERFACE ---

# Custom Header HTML
st.markdown("""
    <div class="main-header">
        <h1>🕉️ Mahabharat GPT</h1>
        <div style="font-style: italic; color: #D35400;">“Wisdom from the Itihasa”</div>
    </div>
""", unsafe_allow_html=True)

if not retriever:
    st.error("⚠️ The Sacred Texts (Database) are missing. Please run the ingestion script first.")
    st.stop()

# Initialize State
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        AIMessage(content="Pranam, seeker of truth. I am the chronicler of the Great War. Ask, and I shall recite from the ancient texts.")
    )

# Handle Sidebar Clicks
if "prompt_input" in st.session_state and st.session_state.prompt_input:
    user_input = st.session_state.prompt_input
    del st.session_state.prompt_input
else:
    user_input = None

# Display History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🙏"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🪔"):
            st.write(msg.content)

# Input Logic
chat_input_val = st.chat_input("Ask about Dharma, Karma, or the War...")
final_query = user_input if user_input else chat_input_val

if final_query:
    # Show User Msg
    if not user_input: 
        with st.chat_message("user", avatar="🙏"):
            st.write(final_query)
    st.session_state.messages.append(HumanMessage(content=final_query))

    # Generate Response
    with st.chat_message("assistant", avatar="🪔"):
        with st.spinner("Meditating on the scriptures..."):
            
            # Retrieve
            docs = retriever.invoke(final_query)
            context_text = "\n\n".join([d.page_content for d in docs])
            
            # Prompt
            system_prompt = (
                "You are Sanjaya, the wise narrator of the Mahabharata. "
                "CONTEXT:\n" + context_text + "\n\n"
                "RULES:"
                "1. LANGUAGE: Match the user's language (English, Hindi, or Hinglish)."
                "2. SCRIPT: If user types in Latin script (e.g. 'Karna kaun tha'), use Latin script (Hinglish). Do NOT use Devanagari unless user does."
                "3. TONE: Respectful, epic, like a Rishi."
                "4. CONTENT: Answer strictly based on context."
            )
            
            messages = [SystemMessage(content=system_prompt)] + st.session_state.messages
            response = llm.invoke(messages)
            
            st.markdown(response.content)
            st.session_state.messages.append(AIMessage(content=response.content))