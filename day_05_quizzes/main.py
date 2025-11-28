import os
from dotenv import load_dotenv
from day_02_reading.loaders import DocumentLoader
from day_03_chunking.chunker import chunk_documents
from day_03_chunking.retriever import create_hybrid_retriever
from day_05_quizzes.generator import generate_quiz

load_dotenv()

# --- setup ---

def create_knowledge_base(path: str):
    """Creates a dummy text file with content for the quiz."""
    content = """
    LangChain is a framework for developing applications powered by language models.
    It enables applications that:
    - Are context-aware: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
    - Reason: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)
    
    The main value props of LangChain are:
    1. Components: composable tools and integrations for working with language models. Components are modular and easy-to-use, whether you are using the rest of the LangChain framework or not.
    2. Off-the-shelf chains: built-in assemblages of components for accomplishing higher-level tasks.
    
    LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. 
    Compared to other LLM frameworks, it offers these core benefits: cycles, controllability, and persistence.
    LangGraph allows you to define flows that involve cycles, essential for most agentic architectures.
    It provides fine-grained control over both the flow and state of your application, crucial for creating reliable agents.
    """
    with open(path, "w") as f:
        f.write(content.strip())
    print(f"--- Created knowledge base: {path} ---")

def cleanup(path: str):
    """Removes the file."""
    if os.path.exists(path):
        os.remove(path)
        print(f"--- Cleaned up file: {path} ---")

# --- execution ---

def main():
    print("\n🧠 Starting Day 5: Quizzes (Verification)...\n")
    
    # 1. setup
    file_path = "day_05_quizzes/knowledge.txt"
    create_knowledge_base(file_path)
    
    try:
        # 2. load
        docs = DocumentLoader.load(file_path)
        print(f"✅ Loaded {len(docs)} raw document(s).")
        
        # 3. chunk
        chunks = chunk_documents(docs)
        print(f"✅ Generated {len(chunks)} semantic chunks.")
        
        # 4. index
        retriever = create_hybrid_retriever(chunks)
        print("✅ Created hybrid retriever.")
        
        # 5. generate quiz
        topic = "LangChain vs LangGraph"
        quiz = generate_quiz(topic, retriever)
        
        # 6. print quiz
        print(f"\n📝 Quiz Topic: {quiz.topic}")
        print("-" * 40)
        
        for i, q in enumerate(quiz.questions, 1):
            print(f"\nQ{i}: {q.question}")
            for option in q.options:
                print(f"   {option}")
            print(f"\n   ✅ Answer: {q.correct_answer}")
            print(f"   ℹ️ Explanation: {q.explanation}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 7. cleanup
        print("\n")
        cleanup(file_path)
        print("\n✅ Day 5 Verification Complete!")

if __name__ == "__main__":
    main()
