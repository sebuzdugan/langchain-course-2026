import os
from dotenv import load_dotenv
from day_02_reading.loaders import DocumentLoader
from day_03_chunking.chunker import chunk_documents
from day_03_chunking.retriever import create_hybrid_retriever
from day_06_flashcards.generator import generate_flashcards

load_dotenv()

# --- setup ---

def create_knowledge_base(path: str):
    """Creates a dummy text file with content for the flashcards."""
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
    
    Semantic Chunking is a technique to split text based on meaning. 
    Instead of splitting by a fixed number of characters, it uses embeddings to find "breakpoints" where the topic changes.
    This results in chunks that are semantically complete, improving retrieval quality.
    
    RAG (Retrieval Augmented Generation) is a technique that combines retrieval and generation.
    First, we retrieve relevant documents from a knowledge base using the user's query.
    Then, we pass those documents as context to an LLM, which generates a response based on that context.
    This allows the LLM to provide accurate, up-to-date information without hallucinating.
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
    print("\n🧠 Starting Day 6: Flashcards (Verification)...\n")
    
    # 1. setup
    file_path = "day_06_flashcards/knowledge.txt"
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
        
        # 5. generate flashcards
        topic = "LangChain Concepts"
        flashcard_set = generate_flashcards(topic, retriever)
        
        # 6. print flashcards
        print(f"\n🗂️  Topic: {flashcard_set.topic}")
        print("-" * 40)
        
        for i, card in enumerate(flashcard_set.cards, 1):
            print(f"\nCard {i}:")
            print(f"   Front: {card.front}")
            print(f"   Back:  {card.back}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 7. cleanup
        print("\n")
        cleanup(file_path)
        print("\n✅ Day 6 Verification Complete!")

if __name__ == "__main__":
    main()
