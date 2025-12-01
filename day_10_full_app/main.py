import streamlit as st
import uuid
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add the project root to sys.path so we can import from other days
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the project root to sys.path so we can import from other days
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from day_08_memory.agent import app
from langchain_core.messages import HumanMessage, AIMessage

# Page Config


# Imports for functionality
from day_02_reading.loaders import DocumentLoader
from day_03_chunking.chunker import chunk_documents
from day_03_chunking.retriever import create_hybrid_retriever
from day_05_quizzes.generator import generate_quiz
from day_06_flashcards.generator import generate_flashcards
from day_07_planning.planner import generate_study_plan
import tempfile

# Session State Initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- Sidebar: Chat & Settings ---
with st.sidebar:
    st.header("🧠 AI Assistant")
    
    # 1. Knowledge Base (Collapsible)
    with st.expander("📁 Knowledge Base", expanded=False):
        uploaded_file = st.file_uploader("Upload PDF/Txt", type=["pdf", "txt"])
        if uploaded_file:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            if st.button("Process File"):
                with st.spinner("Processing..."):
                    docs = DocumentLoader.load(tmp_path)
                    chunks = chunk_documents(docs)
                    retriever = create_hybrid_retriever(chunks)
                    st.session_state.retriever = retriever
                    st.success("✅ Indexed!")
    
    st.divider()

    # 2. Chat Interface
    # Use a scrollable container for chat history
    # height=500px ensures it doesn't grow indefinitely
    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 3. Chat Input
    if prompt := st.chat_input("Ask me anything..."):
        # Display User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Run Agent
        with chat_container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # Retrieve context
                context_str = None
                if st.session_state.retriever:
                    with st.spinner("Searching..."):
                        docs = st.session_state.retriever.invoke(prompt)
                        if docs:
                            context_str = "\n\n".join([d.page_content for d in docs])
                            with st.expander("Context"):
                                st.caption(context_str[:200] + "...")

                # Prepare inputs
                inputs = {"question": prompt}
                if context_str:
                    inputs["context"] = context_str
                
                try:
                    with st.spinner("Thinking..."):
                        for event in app.stream(inputs, config=config, stream_mode="values"):
                            final_state = event
                    
                    if final_state and "answer" in final_state:
                        full_response = final_state["answer"]
                        message_placeholder.markdown(full_response)
                    else:
                        full_response = "I'm not sure."
                        message_placeholder.markdown(full_response)
                        

                except Exception as e:
                    full_response = f"Error: {str(e)}"
                    message_placeholder.error(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Main Area: Learning Tools ---
st.title("🧠 AI Learning Assistant")

tab1, tab2, tab3 = st.tabs(["📝 Quiz", "🗂️ Flashcards", "📅 Plan"])

with tab1:
    st.header("Generate a Quiz")
    col_q1, col_q2 = st.columns([0.7, 0.3])
    with col_q1:
        topic = st.text_input("Topic for Quiz", "LangChain")
    with col_q2:
        st.write("") # Spacer
        st.write("")
        if st.button("Create Quiz", use_container_width=True):
            with st.spinner(f"Generating quiz for {topic}..."):
                # Pass retriever if it exists, else None
                quiz = generate_quiz(topic, st.session_state.retriever)
                st.session_state.current_quiz = quiz

    if "current_quiz" in st.session_state:
        quiz = st.session_state.current_quiz
        st.subheader(f"Quiz: {quiz.topic}")
        
        for i, q in enumerate(quiz.questions):
            st.markdown(f"**{i+1}. {q.question}**")
            user_choice = st.radio(
                "Choose an answer:", 
                q.options, 
                key=f"quiz_q_{i}", 
                index=None,
                label_visibility="collapsed"
            )
            if user_choice:
                if user_choice == q.correct_answer:
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Incorrect. Answer: {q.correct_answer}")
                with st.expander("Explanation"):
                    st.info(q.explanation)
            st.divider()

with tab2:
    st.header("Generate Flashcards")
    col_f1, col_f2 = st.columns([0.7, 0.3])
    with col_f1:
        fc_topic = st.text_input("Topic for Flashcards", "Key Concepts")
    with col_f2:
        st.write("")
        st.write("")
        if st.button("Create Flashcards", use_container_width=True):
            with st.spinner("Generating..."):
                flashcards = generate_flashcards(fc_topic, st.session_state.retriever)
                st.session_state.current_flashcards = flashcards
                
    if "current_flashcards" in st.session_state:
        flashcards = st.session_state.current_flashcards
        st.subheader(f"Flashcards: {flashcards.topic}")
        
        # Grid layout for flashcards
        cols = st.columns(2)
        for i, card in enumerate(flashcards.cards):
            with cols[i % 2]:
                with st.expander(f"📌 {card.front}", expanded=True):
                    st.markdown(f"**{card.back}**")

with tab3:
    st.header("Create Study Plan")
    col_p1, col_p2, col_p3 = st.columns([0.4, 0.3, 0.3])
    with col_p1:
        sp_topic = st.text_input("Topic", "LangChain Agents")
    with col_p2:
        duration = st.text_input("Time", "2 hours")
    with col_p3:
        st.write("")
        st.write("")
        if st.button("Plan Session", use_container_width=True):
            with st.spinner("Planning..."):
                plan = generate_study_plan(sp_topic, duration, st.session_state.retriever)
                st.session_state.current_plan = plan
                
    if "current_plan" in st.session_state:
        plan = st.session_state.current_plan
        st.info(f"**Goal:** {plan.goal}")
        
        for i, session in enumerate(plan.sessions):
            with st.container(border=True):
                st.markdown(f"### 🗓️ Session {i+1}: {session.goal}")
                st.caption(f"Duration: {session.total_duration_minutes} mins")
                for item in session.items:
                    st.markdown(f"- **{item.activity}** ({item.duration_minutes}m)")
