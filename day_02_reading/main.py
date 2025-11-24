import os
from day_02_reading.loaders import DocumentLoader

# --- setup ---

def create_dummy_text_file(path: str):
    """Creates a dummy text file for testing."""
    with open(path, "w") as f:
        f.write("This is a sample text file for Day 2 of LangChain 2026.\n")
        f.write("It contains multiple lines to test loading.\n")
        f.write("LangChain is awesome!")
    print(f"--- Created dummy file: {path} ---")

def cleanup_file(path: str):
    """Removes a file if it exists."""
    if os.path.exists(path):
        os.remove(path)
        print(f"--- Cleaned up file: {path} ---")

# --- execution ---

def main():
    print("\n🧠 Starting Day 2: Reading Sources (Verification)...\n")
    
    # 1. setup
    text_path = "day_02_reading/sample.txt"
    create_dummy_text_file(text_path)
    
    # 2. define sources to test
    sources = [
        text_path,
        "https://python.langchain.com/docs/introduction/"
    ]
    
    # 3. test loaders
    for source in sources:
        print(f"\n🔍 Testing Source: {source}")
        try:
            docs = DocumentLoader.load(source)
            print(f"✅ Successfully loaded {len(docs)} document(s).")
            
            if docs:
                first_doc = docs[0]
                content_preview = first_doc.page_content[:150].replace('\n', ' ')
                print(f"   📄 Metadata: {first_doc.metadata}")
                print(f"   📝 Content Preview: {content_preview}...")
        except Exception as e:
            print(f"❌ Failed to load {source}: {e}")
            
    # 4. cleanup
    print("\n")
    cleanup_file(text_path)
    print("\n✅ Day 2 Verification Complete!")

if __name__ == "__main__":
    main()
