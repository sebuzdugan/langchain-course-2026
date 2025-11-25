import os
import shutil
from dotenv import load_dotenv
from day_02_reading.loaders import DocumentLoader
from day_03_chunking.chunker import chunk_documents
from day_03_chunking.retriever import create_hybrid_retriever

load_dotenv()

# --- setup ---

def create_knowledge_base(path: str):
    """Creates a dummy text file with enough content for chunking."""
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
    
    Semantic Chunking is a technique to split text based on meaning. 
    Instead of splitting by a fixed number of characters, it uses embeddings to find "breakpoints" where the topic changes.
    This results in chunks that are semantically complete, improving retrieval quality.
    """
    with open(path, "w") as f:
        f.write(content.strip())
    print(f"--- Created knowledge base: {path} ---")

def cleanup(path: str):
    """Removes the file and chroma db."""
    if os.path.exists(path):
        os.remove(path)
        print(f"--- Cleaned up file: {path} ---")
    # cleanup chroma if needed (it creates a local folder usually?)
    # default chroma uses in-memory or temp dir unless persisted.
    # we didn't specify persist_directory, so it might be ephemeral or in default location.
    # if it creates a folder, we might want to clean it.
    # for now, we'll leave it as we didn't specify a path.

# --- execution ---

def main():
    print("\n🧠 Starting Day 3: Smart Chunking & Hybrid Retrieval (Verification)...\n")
    
    # 1. setup
    file_path = "day_03_chunking/knowledge.txt"
    create_knowledge_base(file_path)
    
    try:
        # 2. load
        docs = DocumentLoader.load(file_path)
        print(f"✅ Loaded {len(docs)} raw document(s).")
        
        # 3. chunk
        chunks = chunk_documents(docs)
        print(f"✅ Generated {len(chunks)} semantic chunks.")
        
        # 4. index & retrieve
        retriever = create_hybrid_retriever(chunks)
        
        # 5. test query
        query = "What is the difference between LangChain and LangGraph?"
        print(f"\n❓ Query: {query}")
        
        results = retriever.invoke(query)
        print(f"✅ Retrieved {len(results)} relevant chunks.")
        
        for i, doc in enumerate(results):
            print(f"\n   📄 Result {i+1}:")
            content_preview = doc.page_content[:150].replace('\n', ' ')
            print(f"   {content_preview}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # 6. cleanup
        print("\n")
        cleanup(file_path)
        print("\n✅ Day 3 Verification Complete!")

if __name__ == "__main__":
    main()
