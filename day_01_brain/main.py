from day_01_brain.agent import app

def run_agent(question: str):
    print(f"\n👤 User: {question}")
    result = app.invoke({"question": question})
    print(f"🤖 Agent: {result['answer']}")
    print(f"👣 Steps: {result['steps']}")

def main():
    print("🧠 Starting Day 1 Brain (Full Architecture)...")
    
    # test explanation flow
    run_agent("What is LangGraph?")
    
    # test quiz flow
    run_agent("Generate a quiz about vector databases")
    
    # test flashcard flow
    run_agent("Make flashcards for semantic chunking")
    
    # test study plan flow
    run_agent("Create a study plan for learning AI")

if __name__ == "__main__":
    main()
